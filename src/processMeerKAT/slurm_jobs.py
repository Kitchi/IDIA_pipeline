"""SLURM job script generation and path utilities."""

import os
import logging
from copy import deepcopy
from datetime import datetime

from . import config_parser
from .constants import (
    LOG_DIR, SCRIPT_DIR, CALIB_SCRIPTS_DIR, AUX_SCRIPTS_DIR,
    SELFCAL_SCRIPTS_DIR, PIPELINE_STATE, MASTER_SCRIPT, SPW_PREFIX, THIS_PROG,
    MPI_WRAPPER, CONTAINER,
    CPUS_PER_NODE_LIMIT, MEM_PER_NODE_GB_LIMIT, MEM_PER_NODE_GB_LIMIT_HIGHMEM,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def check_bash_path(fname):
    """Check if file is in your bash PATH and executable; prepend path if so."""
    for path in os.environ['PATH'].split(':'):
        full = '{0}/{1}'.format(path, fname)
        if os.path.exists(full):
            if not os.access(full, os.X_OK):
                raise IOError('"{0}" found in "{1}" but file is not executable.'.format(fname, path))
            return full
    return fname


def check_path(path, update=False):
    """Check in specific location for a script or container.

    Searches CWD, parent directory, pipeline script directories, and bash PATH.
    Raises IOError if not found.
    """
    newpath = path

    if os.path.exists(path) and path[0] != '/':
        newpath = '{0}/{1}'.format(os.getcwd(), path)
    if not os.path.exists(path) and path != '':
        if os.path.exists('../{0}'.format(path)):
            newpath = '../{0}'.format(path)
        elif os.path.exists('{0}/{1}'.format(SCRIPT_DIR, path)):
            newpath = '{0}/{1}'.format(SCRIPT_DIR, path)
        elif os.path.exists('{0}/{1}/{2}'.format(SCRIPT_DIR, CALIB_SCRIPTS_DIR, path)):
            newpath = '{0}/{1}/{2}'.format(SCRIPT_DIR, CALIB_SCRIPTS_DIR, path)
        elif os.path.exists('{0}/{1}/{2}'.format(SCRIPT_DIR, AUX_SCRIPTS_DIR, path)):
            newpath = '{0}/{1}/{2}'.format(SCRIPT_DIR, AUX_SCRIPTS_DIR, path)
        elif os.path.exists('{0}/{1}/{2}'.format(SCRIPT_DIR, SELFCAL_SCRIPTS_DIR, path)):
            newpath = '{0}/{1}/{2}'.format(SCRIPT_DIR, SELFCAL_SCRIPTS_DIR, path)
        elif os.path.exists(check_bash_path(path)):
            newpath = check_bash_path(path)
        else:
            raise IOError('File "{0}" not found.'.format(path))

    if update:
        return newpath
    return path


# ---------------------------------------------------------------------------
# srun helper
# ---------------------------------------------------------------------------

def srun(arg_dict, qos=True, time=10, mem=4):
    """Build an srun command string with resource parameters."""
    call = 'srun --time={0} --mem={1}GB --partition={2} --account={3}'.format(
        time, mem, arg_dict['partition'], arg_dict['account']
    )
    if qos:
        call += ' --qos qos-interactive'
    if arg_dict.get('exclude', '') != '':
        call += ' --exclude={0}'.format(arg_dict['exclude'])
    if arg_dict.get('reservation', '') != '':
        call += ' --reservation={0}'.format(arg_dict['reservation'])
    return call


# ---------------------------------------------------------------------------
# Command and sbatch generation
# ---------------------------------------------------------------------------

def _is_cal_partition(script):
    """True only for the per-SPW calibrator partition (not partition_target)."""
    name = os.path.basename(script).split('.')[0]
    return name == 'partition'


# Subdirs of the package that hold runnable scripts. Order matters only when a
# name collides — package root wins last, falling through to subpackages first.
_SCRIPT_SUBPACKAGES = (
    ('crosscal_scripts', 'processMeerKAT.crosscal_scripts'),
    ('selfcal_scripts', 'processMeerKAT.selfcal_scripts'),
    ('aux_scripts', 'processMeerKAT.aux_scripts'),
    ('', 'processMeerKAT'),
)


def script_module_path(script):
    """Map a script name (or path) to a dotted Python module path.

    Returns None if the script isn't found inside the installed package — the
    caller should then fall back to file-path invocation (`python {script}`).
    """
    name = os.path.basename(script)
    base, ext = os.path.splitext(name)
    if ext != '.py':
        return None
    for subdir, pkg in _SCRIPT_SUBPACKAGES:
        candidate = os.path.join(SCRIPT_DIR, subdir, name) if subdir else os.path.join(SCRIPT_DIR, name)
        if os.path.isfile(candidate):
            return f'{pkg}.{base}'
    return None


def _resolve_runner(container, default_runner):
    """Pick the command prefix for a script invocation.

    Per-script container override wins (built as `singularity exec <path>` for
    backward compat with existing configs). Otherwise the facility's
    default_runner is used (which itself may be `singularity exec ...`,
    `conda run -n env ...`, or empty).
    """
    if container:
        return 'singularity exec {0}'.format(container)
    return default_runner or ''


def write_command(script, args, name='job', mpi_wrapper=MPI_WRAPPER,
                  container=CONTAINER, casa_script=False, logfile=True,
                  plot=False, SPWs='', nspw=1, default_runner=''):
    """Write a bash command calling a script.

    The invoker is `python -m processMeerKAT.<subpath>.<module>` for any
    script that ships with the package; user-supplied scripts not inside the
    package fall back to `python {path/to/script}`. `casa --nogui -c` is no
    longer used — CASA 6+ ships its tooling as plain importable Python
    packages, so a direct `python` invocation works in both container and
    bare-env modes.
    """

    arrayJob = ',' in SPWs and _is_cal_partition(script) and nspw > 1

    runner = _resolve_runner(container, default_runner)
    plot_call = 'xvfb-run -a' if plot else ''

    module_path = script_module_path(script)
    if module_path is not None:
        invoke = f'python -m {module_path}'
    else:
        invoke = f'python {check_path(script, update=True)}'

    parts = [p for p in [runner, mpi_wrapper, plot_call, invoke, args] if p]
    line = ' '.join(parts)

    if arrayJob:
        prefix = (
            '#Iterate over SPWs in job array, launching one after the other\n'
            'SPWs="{0}"\n'
            'arr=($SPWs)\n'
            'cd ${{arr[SLURM_ARRAY_TASK_ID]}}\n\n'
        ).format(SPWs.replace(',', ' ').replace(SPW_PREFIX, ''))
        return prefix + line + '\ncd ..\n'
    return line


def write_sbatch(script, args, nodes=1, tasks=16, mem=MEM_PER_NODE_GB_LIMIT,
                 name="job", runname='', plane=1, exclude='',
                 mpi_wrapper=MPI_WRAPPER, container=CONTAINER,
                 partition="Main", time="12:00:00", casa_script=False,
                 SPWs='', nspw=1, account='b03-idia-ag', reservation='',
                 modules=[], justrun=False, default_runner=''):
    """Write a SLURM sbatch file for a single pipeline step."""

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    params = locals()
    params['LOG_DIR'] = LOG_DIR

    params['cpus'] = 1
    if 'tclean' in script or 'selfcal' in script or _is_cal_partition(script) or 'image' in script:
        params['cpus'] = int(CPUS_PER_NODE_LIMIT / tasks)
    if _is_cal_partition(script):
        dopol = config_parser.get_key(PIPELINE_STATE, 'state', 'dopol')
        if dopol and 4 * tasks < CPUS_PER_NODE_LIMIT:
            params['cpus'] = 4
        elif not dopol and params['cpus'] > 2:
            params['cpus'] = 2

    if params['cpus'] * tasks == CPUS_PER_NODE_LIMIT:
        if params['partition'] == 'HighMem':
            params['mem'] = MEM_PER_NODE_GB_LIMIT_HIGHMEM
        else:
            params['mem'] = MEM_PER_NODE_GB_LIMIT

    plot = ('plot' in script)
    if script == 'validate_input.py':
        casa_script = False
    elif 'bdsf' in script or 'column' in script:
        casa_script = False

    nconcurrent = int(200 / (params['nodes'] * params['tasks'] * params['cpus']))
    if nconcurrent > nspw:
        nconcurrent = nspw

    params['command'] = write_command(
        script, args, name=name, mpi_wrapper=mpi_wrapper,
        container=container, casa_script=casa_script, plot=plot,
        SPWs=SPWs, nspw=nspw, default_runner=default_runner,
    )
    if _is_cal_partition(script) and ',' in SPWs and nspw > 1:
        params['ID'] = '%A_%a'
        params['array'] = '\n#SBATCH --array=0-{0}%{1}'.format(nspw - 1, nconcurrent)
    else:
        params['ID'] = '%j'
        params['array'] = ''
    params['exclude'] = '\n#SBATCH --exclude={0}'.format(exclude) if exclude != '' else ''
    params['reservation'] = '\n#SBATCH --reservation={0}'.format(reservation) if reservation != '' else ''

    if 'selfcal' in script or 'image' in script:
        params['command'] = 'ulimit -n 16384\n' + params['command']

    params['modules'] = ''
    for module in modules:
        if len(module) > 0:
            params['modules'] += "module load {0}\n".format(module)

    contents = """#!/bin/bash{array}{exclude}{reservation}
    #SBATCH --account={account}
    #SBATCH --nodes={nodes}
    #SBATCH --ntasks-per-node={tasks}
    #SBATCH --cpus-per-task={cpus}
    #SBATCH --mem={mem}GB
    #SBATCH --job-name={runname}{name}
    #SBATCH --distribution=plane={plane}
    #SBATCH --output={LOG_DIR}/%x-{ID}.out
    #SBATCH --error={LOG_DIR}/%x-{ID}.err
    #SBATCH --partition={partition}
    #SBATCH --time={time}

    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    {modules}

    {command}"""

    contents = contents.format(**params).replace("    ", "")

    sbatch = '{0}.sbatch'.format(name)
    if justrun and os.path.exists(sbatch):
        logger.debug('sbatch file "{0}" exists. Not overwriting due to --justrun.'.format(sbatch))
    else:
        with open(sbatch, 'w') as f:
            f.write(contents)
        logger.debug('Wrote sbatch file "{0}"'.format(sbatch))


# ---------------------------------------------------------------------------
# Bash utility script generation
# ---------------------------------------------------------------------------

def write_bash_job_script(master, filename, extn, do, purpose,
                          dir='jobScripts', echo=True, prefix=''):
    """Write a single bash utility script (kill, summary, errors, etc.)."""
    fname = '{0}/{1}{2}'.format(dir, filename, extn)
    do2 = ' ./{0}/{1}{2}{3} \\$@"'.format(dir, prefix, filename, extn) if prefix != '' else ' '
    master.write('\n#Create {0}.sh file, make executable and symlink to current version\n'.format(filename))
    master.write('echo "#!/bin/bash" > {0}\n'.format(fname))
    master.write('{0}{1}>> {2}\n'.format(do, do2, fname))
    master.write('chmod +x {0}\n'.format(fname))
    master.write('ln -f -s {0} {1}.sh\n'.format(fname, filename))
    if echo:
        master.write('echo "Run ./{0}.sh to {1}."\n'.format(filename, purpose))


def write_all_bash_jobs_scripts(master, extn, IDs, dir='jobScripts',
                                echo=True, prefix='', pad_length=5,
                                slurm_kwargs={}):
    """Write all ancillary bash scripts (kill, summary, errors, timing, cleanup)."""
    killScript = prefix + 'killJobs'
    summaryScript = prefix + 'summary'
    errorScript = prefix + 'findErrors'
    timingScript = prefix + 'displayTimes'
    cleanupScript = prefix + 'cleanup'

    write_bash_job_script(
        master, killScript, extn,
        'echo scancel ${0}'.format(IDs),
        'kill all the jobs', dir=dir, echo=echo,
    )
    do = (
        'echo sacct -j ${0} --units=G -o '
        '"JobID%-15,JobName%-{1},Partition,Elapsed,NNodes%6,NTasks%6,'
        'NCPUS%5,MaxDiskRead,MaxDiskWrite,NodeList%20,TotalCPU,CPUTime,'
        'MaxRSS,State,ExitCode" \\$@ '.format(IDs, 15 + pad_length)
    )
    write_bash_job_script(master, summaryScript, extn, do, 'view the progress', dir=dir, echo=echo)

    do = (
        'echo "for ID in {{${IDS},}}; do '
        'files=\\$(ls {LD}/*\\$ID* 2>/dev/null | wc -l); '
        'if [ \\$((files)) != 0 ]; then '
        'ls {LD}/*\\$ID*; '
        "cat {LD}/*\\$ID* | grep -i 'severe\\|error' | "
        "grep -vi 'mpi\\|The selected table has zero rows\\|MeasTable::dUTC(Double)'; "
        "else echo {LD}/*\\$ID* logs don\\'t exist \\(yet\\); "
        'fi; done" '
    ).format(IDS=IDs, LD=LOG_DIR)
    write_bash_job_script(master, errorScript, extn, do, 'find errors (after pipeline has run)', dir=dir, echo=echo)

    do = (
        'echo "for ID in {{${IDS},}}; do '
        'files=\\$(ls {LD}/*\\$ID* 2>/dev/null | wc -l); '
        'if [ \\$((files)) != 0 ]; then '
        'logs=\\$(ls {LD}/*\\$ID* | sort -V); ls -f \\$logs; '
        "cat \\$(ls -tU \\$logs) | grep INFO | head -n 1 | cut -d 'I' -f1; "
        "cat \\$(ls -tr \\$logs) | grep INFO | tail -n 1 | cut -d 'I' -f1; "
        "else echo {LD}/*\\$ID* logs don\\'t exist \\(yet\\); "
        'fi; done" '
    ).format(IDS=IDs, LD=LOG_DIR)
    write_bash_job_script(master, timingScript, extn, do, 'display start and end timestamps (after pipeline has run)', dir=dir, echo=echo)

    cleanup_kwargs = deepcopy(slurm_kwargs)
    cleanup_kwargs['partition'] = 'Devel'
    do = (
        'echo "echo Removing the following: \\$(ls -d *ms); {srun} rm -r *ms" '
    ).format(srun=srun(cleanup_kwargs, qos=True, time=10, mem=0))
    write_bash_job_script(master, cleanupScript, extn, do, 'remove MSs/MMSs from this directory (after pipeline has run)', dir=dir, echo=echo)


# ---------------------------------------------------------------------------
# Master script generation helpers
# ---------------------------------------------------------------------------

def _expand_selfcal_loops(config, scripts):
    """Expand selfcal_part1/2 in script list to cover all selfcal loops."""
    if not (config_parser.has_section(config, 'selfcal')
            and 'selfcal_part1.sbatch' in scripts
            and 'selfcal_part2.sbatch' in scripts):
        return scripts
    start_loop = config_parser.get_key(config, 'selfcal', 'loop')
    selfcal_loops = config_parser.get_key(config, 'selfcal', 'nloops') - start_loop
    idx = scripts.index('selfcal_part2.sbatch')
    if idx != scripts.index('selfcal_part1.sbatch') + 1:
        return scripts
    head = scripts[:idx + 1]
    tail = scripts[idx + 1:]
    head.extend(['selfcal_part1.sbatch', 'selfcal_part2.sbatch'] * (selfcal_loops - 1))
    head.append('selfcal_part1.sbatch')
    return head + tail


# ---------------------------------------------------------------------------
# Master script generation
# ---------------------------------------------------------------------------

def write_master(filename, config, scripts=[], submit=False,
                 dir='jobScripts', pad_length=5, verbose=False,
                 echo=True, dependencies='', slurm_kwargs={}):
    """Write master pipeline submission script (single-SPW case)."""

    timestamp = config_parser.get_key(config, 'state', 'timestamp')
    if timestamp == '':
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        config_parser.overwrite_config(
            config,
            conf_dict={'timestamp': timestamp},
            conf_sec='state',
        )

    scripts = _expand_selfcal_loops(config, scripts)

    master = open(filename, 'w')
    master.write('#!/bin/bash\n')
    if verbose:
        master.write("\necho Copying '{0}' to '{1}', and using this to run pipeline.\n".format(config, PIPELINE_STATE))
    master.write('cp {0} {1}\n'.format(config, PIPELINE_STATE))

    command = 'sbatch'
    if dependencies != '':
        master.write('\n#Run after these dependencies\nDep={0}\n'.format(dependencies))
        command += " -d afterok:${Dep//,/:} --kill-on-invalid-dep=yes"
    master.write('\n#{0}\n'.format(scripts[0]))
    if verbose:
        master.write('echo Submitting {0} to SLURM queue with following command:\necho {1} {0}.\n'.format(scripts[0], command))
    master.write("IDs=$({0} {1} | cut -d ' ' -f4)\n".format(command, scripts[0]))
    scripts.pop(0)

    for script in scripts:
        command = "sbatch -d afterok:${IDs//,/:} --kill-on-invalid-dep=yes"
        master.write('\n#{0}\n'.format(script))
        if verbose:
            master.write('echo Submitting {0} to SLURM queue with following command\necho {1} {0}.\n'.format(script, command))
        master.write("IDs+=,$({0} {1} | cut -d ' ' -f4)\n".format(command, script))

    master.write('\n#Output message and create {0} directory\n'.format(dir))
    master.write('echo Submitted sbatch jobs with following IDs: $IDs\n')
    master.write('echo "$IDs" > submitted_jobs.txt\n')
    master.write('mkdir -p {0}\n'.format(dir))

    master.write('\n#Add time as extn to this pipeline run, to give unique filenames')
    master.write("\nDATE={0}".format(timestamp))
    extn = '_$DATE.sh'

    master.write('\n#Copy contents of config file to {0} directory\n'.format(dir))
    master.write('cp {0} {1}/{2}_$DATE.txt\n'.format(config, dir, os.path.splitext(config)[0]))

    write_all_bash_jobs_scripts(master, extn, IDs='IDs', dir=dir, echo=echo,
                                pad_length=pad_length, slurm_kwargs=slurm_kwargs)

    master.close()
    os.chmod(filename, 509)

    if submit:
        if echo:
            logger.info('Running master script "{0}"'.format(filename))
        os.system('./{0}'.format(filename))
    else:
        logger.info('Master script "{0}" written in "{1}", but will not run.'.format(
            filename, os.path.split(os.getcwd())[-1]
        ))


def write_spw_master(filename, config, SPWs, precal_scripts, postcal_scripts,
                     submit, dir='jobScripts', pad_length=5, dependencies='',
                     timestamp='', slurm_kwargs={}, target_scripts=None):
    """Write top-level master script for multi-SPW pipeline.

    DAG: precal_scripts run once at top level (chained). The cal partition is
    a job-array (one element per SPW) feeding parallel per-SPW solve chains.
    If target_scripts and a `partition_target.sbatch` precal entry are present,
    a parallel target branch is emitted depending on partition_target's job ID.
    postcal_scripts run after both branches join.
    """

    target_scripts = target_scripts or []

    master = open(filename, 'w')
    master.write('#!/bin/bash\n')
    SPWs = SPWs.replace(SPW_PREFIX, '')
    toplevel = len(precal_scripts + postcal_scripts) > 0

    scripts = precal_scripts[:]
    if len(scripts) > 0:
        command = 'sbatch'
        if dependencies != '':
            master.write('\n#Run after these dependencies\nDep={0}\n'.format(dependencies))
            command += " -d afterok:${Dep//,/:} --kill-on-invalid-dep=yes"
            dependencies = ''
        master.write('\n#{0}\n'.format(scripts[0]))
        master.write("allSPWIDs=$({0} {1} | cut -d ' ' -f4)\n".format(command, scripts[0]))
        scripts.pop(0)
    for script in scripts:
        command = "sbatch -d afterok:${allSPWIDs//,/:} --kill-on-invalid-dep=yes"
        master.write('\n#{0}\n'.format(script))
        master.write("allSPWIDs+=,$({0} {1} | cut -d ' ' -f4)\n".format(command, script))

    if 'calc_refant.sbatch' in precal_scripts:
        master.write('echo Calculating reference antenna, and copying result to SPW directories.\n')
    if 'partition.sbatch' in precal_scripts:
        master.write('echo Running partition job array, iterating over {0} SPWs.\n'.format(len(SPWs.split(','))))
    if 'partition_target.sbatch' in precal_scripts:
        master.write('echo Splitting target into a single MMS for the parallel target branch.\n')

    # Locate partition + partition_target job IDs by name (1-indexed cut field).
    partition_idx = (precal_scripts.index('partition.sbatch') + 1
                     if 'partition.sbatch' in precal_scripts else None)
    target_partition_idx = (precal_scripts.index('partition_target.sbatch') + 1
                            if 'partition_target.sbatch' in precal_scripts else None)
    partition = partition_idx is not None
    if partition:
        master.write('\npartitionID=$(echo $allSPWIDs | cut -d , -f{0})\n'.format(partition_idx))
    if target_partition_idx is not None:
        master.write('targetPartID=$(echo $allSPWIDs | cut -d , -f{0})\n'.format(target_partition_idx))

    killScript = 'killJobs'
    summaryScript = 'summary'
    fullSummaryScript = 'fullSummary'
    errorScript = 'findErrors'
    timingScript = 'displayTimes'
    cleanupScript = 'cleanup'

    master.write('\n#Add time as extn to this pipeline run, to give unique filenames')
    master.write("\nDATE={0}\n".format(timestamp))
    master.write('mkdir -p {0}\n'.format(dir))
    master.write('mkdir -p {0}\n\n'.format(LOG_DIR))
    extn = '_$DATE.sh'

    prog_name = os.path.split(THIS_PROG)[1]
    for i, spw in enumerate(SPWs.split(',')):
        master.write('echo Running pipeline in directory "{1}" for spectral window {0}{1}\n'.format(SPW_PREFIX, spw))
        master.write('cd {0}\n'.format(spw))
        master.write('output=$({0} --config ./{1} --run --submit --quiet --justrun'.format(prog_name, config))
        if partition:
            master.write(' --dependencies=$partitionID\\_{0}'.format(i))
        elif len(precal_scripts) > 0:
            master.write(' --dependencies=$allSPWIDs')
        elif dependencies != '':
            master.write(' --dependencies={0}'.format(dependencies))
        master.write(')\necho -e $output\n')
        if i == 0:
            master.write("IDs=$(echo $output | sed 's/.*IDs\\:\\s\\(.*\\)/\\1/')")
        else:
            master.write("IDs+=,$(echo $output | sed 's/.*IDs\\:\\s\\(.*\\)/\\1/')")
        master.write('\ncd ..\n\n')

    # Target branch — runs in parallel with the per-SPW solve chains.
    target_branch_active = bool(target_scripts) and target_partition_idx is not None
    if target_branch_active:
        master.write('\n#Target branch (parallel to per-SPW calibrator chains)\n')
        first = True
        for script in target_scripts:
            if first:
                command = "sbatch -d afterok:${targetPartID//,/:} --kill-on-invalid-dep=yes"
                master.write('\n#{0}\n'.format(script))
                master.write("targetIDs=$({0} {1} | cut -d ' ' -f4)\n".format(command, script))
                first = False
            else:
                command = "sbatch -d afterok:${targetIDs//,/:} --kill-on-invalid-dep=yes"
                master.write('\n#{0}\n'.format(script))
                master.write("targetIDs+=,$({0} {1} | cut -d ' ' -f4)\n".format(command, script))

    if 'concat.sbatch' in postcal_scripts:
        master.write('echo Will concatenate MSs/MMSs and create quick-look continuum cube across all SPWs for all fields from "{0}".\n'.format(config))
    scripts = _expand_selfcal_loops(config, postcal_scripts[:])

    # Postcal first script joins both branches (per-SPW chains and target branch).
    join_dep = "${IDs//,/:}"
    if target_branch_active:
        join_dep += ":${targetIDs//,/:}"

    if len(scripts) > 0:
        command = "sbatch -d afterany:" + join_dep
        master.write('\n#{0}\n'.format(scripts[0]))
        if len(precal_scripts) == 0:
            master.write("allSPWIDs=$({0} {1} | cut -d ' ' -f4)\n".format(command, scripts[0]))
        else:
            master.write("allSPWIDs+=,$({0} {1} | cut -d ' ' -f4)\n".format(command, scripts[0]))
        scripts.pop(0)
        for script in scripts:
            command = "sbatch -d afterok:${allSPWIDs//,/:} --kill-on-invalid-dep=yes"
            master.write('\n#{0}\n'.format(script))
            master.write("allSPWIDs+=,$({0} {1} | cut -d ' ' -f4)\n".format(command, script))

    master.write('\necho Submitted the following jobIDs within the {0} SPW directories: $IDs\n'.format(len(SPWs.split(','))))

    prefix = ''
    if toplevel:
        master.write('\necho Submitted the following jobIDs over all SPWs: $allSPWIDs\n')
        master.write('\necho For jobs over all SPWs:\n')
        prefix = 'allSPW_'
        write_all_bash_jobs_scripts(master, extn, IDs='allSPWIDs', dir=dir,
                                    prefix=prefix, pad_length=pad_length,
                                    slurm_kwargs=slurm_kwargs)
        master.write('\nln -f -s {1}{2}{3} {0}/{1}{4}{3}\n'.format(
            dir, prefix, summaryScript, extn, fullSummaryScript
        ))

    master.write('\necho For all jobs within the {0} SPW directories:\n'.format(len(SPWs.split(','))))
    header = '-' * (109 + pad_length)

    close = '' if toplevel else '"'
    do = 'echo "for f in {{{spws},}}; do if [ -d \\$f ]; then cd \\$f; ./{dir}/{ks}{extn}; cd ..; else echo Directory \\$f doesn\'t exist; fi; done;{close}'.format(
        spws=SPWs, dir=dir, ks=killScript, extn=extn, close=close
    )
    write_bash_job_script(master, killScript, extn, do, 'kill all the jobs', dir=dir, prefix=prefix)

    do = 'echo "for f in {{{spws},}}; do if [ -d \\$f ]; then cd \\$f; ./{dir}/{cs}{extn}; cd ..; else echo Directory \\$f doesn\'t exist; fi; done;"'.format(
        spws=SPWs, dir=dir, cs=cleanupScript, extn=extn
    )
    write_bash_job_script(master, cleanupScript, extn, do, 'remove the MMSs/MSs within SPW directories (after pipeline has run)', dir=dir)

    do_tmpl = ("""echo "counter=1; for f in {{{spws},}}; do """
               """echo -n SPW \\#\\$counter:; echo -n \\' \\'; """
               """if [ -d \\$f ]; then cd \\$f; pwd; ./{dir}/{ss}{extn} {extra}; """
               """cd ..; else echo Directory \\$f doesn\\'t exist; fi; """
               """counter=\\$((counter+1)); echo '{hdr}'; done; """)
    do = do_tmpl.format(
        spws=SPWs, dir=dir, ss=summaryScript, extn=extn,
        extra="\\$@ | grep -v 'PENDING\\|COMPLETED'", hdr=header
    )
    if toplevel:
        do += "echo -n 'All SPWs: '; pwd; "
    else:
        do += '"'
    write_bash_job_script(master, summaryScript, extn, do, 'view the progress (for running or failed jobs)', dir=dir, prefix=prefix)

    do = do_tmpl.format(
        spws=SPWs, dir=dir, ss=summaryScript, extn=extn,
        extra='\\$@', hdr=header
    )
    if toplevel:
        do += "echo -n 'All SPWs: '; pwd; "
    else:
        do += '"'
    write_bash_job_script(master, fullSummaryScript, extn, do, 'view the progress (for all jobs)', dir=dir, prefix=prefix)

    header = '-' * (90 + pad_length)
    do = do_tmpl.format(
        spws=SPWs, dir=dir, ss=errorScript, extn=extn, extra='', hdr=header
    )
    if toplevel:
        do += "echo -n 'All SPWs: '; pwd; "
    else:
        do += '"'
    write_bash_job_script(master, errorScript, extn, do, 'find errors (after pipeline has run)', dir=dir, prefix=prefix)

    do = do_tmpl.format(
        spws=SPWs, dir=dir, ss=timingScript, extn=extn, extra='', hdr=header
    )
    if toplevel:
        do += "echo -n 'All SPWs: '; pwd; "
    else:
        do += '"'
    write_bash_job_script(master, timingScript, extn, do, 'display start and end timestamps (after pipeline has run)', dir=dir, prefix=prefix)

    id_vars = ['echo "$allSPWIDs"', 'echo "$IDs"']
    if target_branch_active:
        id_vars.append('echo "$targetIDs"')
    master.write('\n{ ' + '; '.join(id_vars) + '; } > submitted_jobs.txt\n')
    master.close()
    os.chmod(filename, 509)

    # Run each SPW subdir to pre-generate sbatch files (can be edited before submission)
    SPW_run_file = 'out.tmp'
    prog_name = os.path.split(THIS_PROG)[1]
    SPW_run_call = (
        "for f in {{{spws},}}; do if [ -d $f ]; then cd $f; "
        "{prog} --config ./{cfg} --run --quiet; cd ..; "
        "else echo Directory $f doesn\\'t exist; fi; done"
    ).format(spws=','.join(SPWs.split(',')), prog=prog_name, cfg=config)
    with open(SPW_run_file, 'w') as out:
        out.write(SPW_run_call)
    os.system('bash {0}'.format(SPW_run_file))
    os.remove(SPW_run_file)

    if submit:
        logger.info('Running master script "{0}"'.format(filename))
        os.system('./{0}'.format(filename))
    else:
        logger.info('Master script "{0}" written in "{1}", but will not run.'.format(
            filename, os.path.split(os.getcwd())[1]
        ))


# ---------------------------------------------------------------------------
# Top-level job orchestration
# ---------------------------------------------------------------------------

def write_jobs(config, scripts=[], threadsafe=[], containers=[],
               num_precal_scripts=0, mpi_wrapper=MPI_WRAPPER,
               nodes=8, ntasks_per_node=4, mem=MEM_PER_NODE_GB_LIMIT,
               plane=1, partition='Main', time='12:00:00', submit=False,
               name='', verbose=False, quiet=False, dependencies='',
               exclude='', account='b03-idia-ag', reservation='',
               modules=[], timestamp='', justrun=False, target_scripts=None):
    """Write all sbatch files and the master submission script."""

    from .constants import CROSSCAL_CONFIG_KEYS
    from .pipeline import get_config_kwargs
    from .processMeerKAT import _FACILITY

    kwargs = locals()
    crosscal_kwargs = get_config_kwargs(config, 'crosscal', CROSSCAL_CONFIG_KEYS)
    pad_length = len(name)
    target_scripts = target_scripts or []
    default_runner = getattr(_FACILITY, 'default_runner', '')

    for i, script in enumerate(scripts):
        jobname = os.path.splitext(os.path.split(script)[1])[0]
        ts = threadsafe[i]
        write_sbatch(
            script, '--config {0}'.format(PIPELINE_STATE),
            nodes=nodes if ts else 1,
            tasks=ntasks_per_node if ts else 1,
            mem=mem,
            plane=plane if ts else 1,
            mpi_wrapper=mpi_wrapper if ts else 'srun',
            container=containers[i], partition=partition, time=time,
            name=jobname, runname=name,
            SPWs=crosscal_kwargs['spw'], nspw=crosscal_kwargs['nspw'],
            exclude=exclude, account=account, reservation=reservation,
            modules=modules, justrun=justrun, default_runner=default_runner,
        )

    # Target branch: same scripts can appear here under a `target_` prefix and
    # are pointed at `target_config.toml` (written by partition_target.py at
    # runtime) so they read the monolithic target MMS instead of the cal MMSes.
    target_sbatch_names = []
    for entry in target_scripts:
        script_path = entry['script']
        ts = entry['mpi']
        ctr = entry.get('container', '')
        jobname = 'target_' + os.path.splitext(os.path.split(script_path)[1])[0]
        write_sbatch(
            script_path, '--config target_config.toml',
            nodes=nodes if ts else 1,
            tasks=ntasks_per_node if ts else 1,
            mem=mem,
            plane=plane if ts else 1,
            mpi_wrapper=mpi_wrapper if ts else 'srun',
            container=ctr or '', partition=partition, time=time,
            name=jobname, runname=name,
            SPWs=crosscal_kwargs['spw'], nspw=crosscal_kwargs['nspw'],
            exclude=exclude, account=account, reservation=reservation,
            modules=modules, justrun=justrun, default_runner=default_runner,
        )
        target_sbatch_names.append(jobname + '.sbatch')

    scripts = [os.path.split(scripts[i])[1].replace('.py', '.sbatch') for i in range(len(scripts))]
    precal_scripts = scripts[:num_precal_scripts]
    postcal_scripts = scripts[num_precal_scripts:]
    echo = not quiet

    if crosscal_kwargs['nspw'] > 1:
        write_spw_master(
            MASTER_SCRIPT, config,
            SPWs=crosscal_kwargs['spw'],
            precal_scripts=precal_scripts,
            postcal_scripts=postcal_scripts,
            target_scripts=target_sbatch_names,
            submit=submit, pad_length=pad_length,
            dependencies=dependencies, timestamp=timestamp,
            slurm_kwargs=kwargs,
        )
    else:
        write_master(
            MASTER_SCRIPT, config,
            scripts=scripts, submit=submit, pad_length=pad_length,
            verbose=verbose, echo=echo, dependencies=dependencies,
            slurm_kwargs=kwargs,
        )
