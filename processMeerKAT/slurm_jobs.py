"""SLURM job script generation and path utilities."""

import os
import logging
from copy import deepcopy
from datetime import datetime

import config_parser
from constants import (
    LOG_DIR, SCRIPT_DIR, CALIB_SCRIPTS_DIR, AUX_SCRIPTS_DIR,
    SELFCAL_SCRIPTS_DIR, TMP_CONFIG, MASTER_SCRIPT, SPW_PREFIX, THIS_PROG,
    MPI_WRAPPER, CONTAINER,
    CPUS_PER_NODE_LIMIT, MEM_PER_NODE_GB_LIMIT, MEM_PER_NODE_GB_LIMIT_HIGHMEM,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def check_bash_path(fname):
    """Check if file is in your bash PATH and executable; prepend path if so."""
    PATH = os.environ['PATH'].split(':')
    for path in PATH:
        if os.path.exists('{0}/{1}'.format(path, fname)):
            if not os.access('{0}/{1}'.format(path, fname), os.X_OK):
                raise IOError('"{0}" found in "{1}" but file is not executable.'.format(fname, path))
            fname = '{0}/{1}'.format(path, fname)
            break
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

def write_command(script, args, name='job', mpi_wrapper=MPI_WRAPPER,
                  container=CONTAINER, casa_script=False, logfile=True,
                  plot=False, SPWs='', nspw=1):
    """Write a bash command calling a script, optionally via CASA/singularity."""

    arrayJob = ',' in SPWs and 'partition' in script and nspw > 1

    params = locals()
    params['LOG_DIR'] = LOG_DIR
    params['job'] = (
        '${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}' if arrayJob
        else '${SLURM_JOB_ID}'
    )
    params['job'] = '${SLURM_JOB_NAME}-' + params['job']
    params['casa_call'] = ''
    params['casa_log'] = '--nologfile'
    params['plot_call'] = ''
    command = ''

    params['script'] = check_path(script, update=True)

    if plot:
        params['plot_call'] = 'xvfb-run -a'
    if logfile:
        params['casa_log'] = '--logfile {LOG_DIR}/{job}.casa'.format(**params)
    if casa_script:
        params['casa_call'] = "casa --nologger --nogui {casa_log} -c".format(**params)
    else:
        params['casa_call'] = 'python'

    if arrayJob:
        command += """#Iterate over SPWs in job array, launching one after the other
        SPWs="%s"
        arr=($SPWs)
        cd ${arr[SLURM_ARRAY_TASK_ID]}

        """ % SPWs.replace(',', ' ').replace(SPW_PREFIX, '')

    command += "{mpi_wrapper} singularity exec {container} {plot_call} {casa_call} {script} {args}".format(**params)

    if arrayJob:
        command += '\ncd ..\n'

    return command


def write_sbatch(script, args, nodes=1, tasks=16, mem=MEM_PER_NODE_GB_LIMIT,
                 name="job", runname='', plane=1, exclude='',
                 mpi_wrapper=MPI_WRAPPER, container=CONTAINER,
                 partition="Main", time="12:00:00", casa_script=False,
                 SPWs='', nspw=1, account='b03-idia-ag', reservation='',
                 modules=[], justrun=False):
    """Write a SLURM sbatch file for a single pipeline step."""

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    params = locals()
    params['LOG_DIR'] = LOG_DIR

    params['cpus'] = 1
    if 'tclean' in script or 'selfcal' in script or 'partition' in script or 'image' in script:
        params['cpus'] = int(CPUS_PER_NODE_LIMIT / tasks)
    if 'partition' in script:
        dopol = config_parser.get_key(TMP_CONFIG, 'run', 'dopol')
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
        SPWs=SPWs, nspw=nspw,
    )
    if 'partition' in script and ',' in SPWs and nspw > 1:
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
    do2 = ' ./{0}/{1}{2}{3} \\$@ \\"'.format(dir, prefix, filename, extn) if prefix != '' else ' '
    master.write('\n#Create {0}.sh file, make executable and symlink to current version\n'.format(filename))
    master.write('echo "#!/bin/bash" > {0}\n'.format(fname))
    master.write('{0}{1}>> {2}\n'.format(do, do2, fname))
    master.write('chmod +x {0}\n'.format(fname))
    master.write('ln -f -s {0} {1}.sh\n'.format(fname, filename))
    if echo:
        master.write('echo Run ./{0}.sh to {1}.\n'.format(filename, purpose))


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
# Master script generation
# ---------------------------------------------------------------------------

def write_master(filename, config, scripts=[], submit=False,
                 dir='jobScripts', pad_length=5, verbose=False,
                 echo=True, dependencies='', slurm_kwargs={}):
    """Write master pipeline submission script (single-SPW case)."""

    master = open(filename, 'w')
    master.write('#!/bin/bash\n')
    timestamp = config_parser.get_key(config, 'run', 'timestamp')
    if timestamp == '':
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        config_parser.overwrite_config(
            config,
            conf_dict={'timestamp': "'{0}'".format(timestamp)},
            conf_sec='run',
            sec_comment='# Internal variables for pipeline execution',
        )

    if verbose:
        master.write("\necho Copying '{0}' to '{1}', and using this to run pipeline.\n".format(config, TMP_CONFIG))
    master.write('cp {0} {1}\n'.format(config, TMP_CONFIG))

    # Expand selfcal loops if needed
    if (config_parser.has_section(config, 'selfcal')
            and 'selfcal_part1.sbatch' in scripts
            and 'selfcal_part2.sbatch' in scripts):
        start_loop = config_parser.get_key(config, 'selfcal', 'loop')
        selfcal_loops = config_parser.get_key(config, 'selfcal', 'nloops') - start_loop
        idx = scripts.index('selfcal_part2.sbatch')
        if idx == scripts.index('selfcal_part1.sbatch') + 1:
            init_scripts = scripts[:idx + 1]
            final_scripts = scripts[idx + 1:]
            init_scripts.extend(['selfcal_part1.sbatch', 'selfcal_part2.sbatch'] * (selfcal_loops - 1))
            init_scripts.append('selfcal_part1.sbatch')
            scripts = init_scripts + final_scripts

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
                     timestamp='', slurm_kwargs={}):
    """Write top-level master script for multi-SPW pipeline."""

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

    partition = len(precal_scripts) > 0 and 'partition' in precal_scripts[-1]
    if partition:
        master.write('\npartitionID=$(echo $allSPWIDs | cut -d , -f{0})\n'.format(len(precal_scripts)))

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

    if 'concat.sbatch' in postcal_scripts:
        master.write('echo Will concatenate MSs/MMSs and create quick-look continuum cube across all SPWs for all fields from "{0}".\n'.format(config))
    scripts = postcal_scripts[:]

    # Expand selfcal loops if needed
    if (config_parser.has_section(config, 'selfcal')
            and 'selfcal_part1.sbatch' in scripts
            and 'selfcal_part2.sbatch' in scripts):
        start_loop = config_parser.get_key(config, 'selfcal', 'loop')
        selfcal_loops = config_parser.get_key(config, 'selfcal', 'nloops') - start_loop
        idx = scripts.index('selfcal_part2.sbatch')
        if idx == scripts.index('selfcal_part1.sbatch') + 1:
            init_scripts = scripts[:idx + 1]
            final_scripts = scripts[idx + 1:]
            init_scripts.extend(['selfcal_part1.sbatch', 'selfcal_part2.sbatch'] * (selfcal_loops - 1))
            init_scripts.append('selfcal_part1.sbatch')
            scripts = init_scripts + final_scripts

    if len(scripts) > 0:
        command = "sbatch -d afterany:${IDs//,/:}"
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

    do = """echo "for f in {{{spws},}}; do if [ -d \\$f ]; then cd \\$f; ./{dir}/{ks}{extn}; cd ..; else echo Directory \\$f doesn\\'t exist; fi; done;{suffix}""".format(
        spws=SPWs, dir=dir, ks=killScript, extn=extn,
        suffix='' if toplevel else ' \\"'
    )
    write_bash_job_script(master, killScript, extn, do, 'kill all the jobs', dir=dir, prefix=prefix)

    do = """echo "for f in {{{spws},}}; do if [ -d \\$f ]; then cd \\$f; ./{dir}/{cs}{extn}; cd ..; else echo Directory \\$f doesn\\'t exist; fi; done; \\\"""".format(
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
        do += ' \\"'
    write_bash_job_script(master, summaryScript, extn, do, 'view the progress (for running or failed jobs)', dir=dir, prefix=prefix)

    do = do_tmpl.format(
        spws=SPWs, dir=dir, ss=summaryScript, extn=extn,
        extra='\\$@', hdr=header
    )
    if toplevel:
        do += "echo -n 'All SPWs: '; pwd; "
    else:
        do += ' \\"'
    write_bash_job_script(master, fullSummaryScript, extn, do, 'view the progress (for all jobs)', dir=dir, prefix=prefix)

    header = '-' * (90 + pad_length)
    do = do_tmpl.format(
        spws=SPWs, dir=dir, ss=errorScript, extn=extn, extra='', hdr=header
    )
    if toplevel:
        do += "echo -n 'All SPWs: '; pwd; "
    else:
        do += ' \\"'
    write_bash_job_script(master, errorScript, extn, do, 'find errors (after pipeline has run)', dir=dir, prefix=prefix)

    do = do_tmpl.format(
        spws=SPWs, dir=dir, ss=timingScript, extn=extn, extra='', hdr=header
    )
    if toplevel:
        do += "echo -n 'All SPWs: '; pwd; "
    else:
        do += ' \\"'
    write_bash_job_script(master, timingScript, extn, do, 'display start and end timestamps (after pipeline has run)', dir=dir, prefix=prefix)

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
               modules=[], timestamp='', justrun=False):
    """Write all sbatch files and the master submission script."""

    from constants import CROSSCAL_CONFIG_KEYS
    from pipeline import get_config_kwargs

    kwargs = locals()
    crosscal_kwargs = get_config_kwargs(config, 'crosscal', CROSSCAL_CONFIG_KEYS)
    pad_length = len(name)

    for i, script in enumerate(scripts):
        jobname = os.path.splitext(os.path.split(script)[1])[0]
        if threadsafe[i]:
            write_sbatch(
                script, '--config {0}'.format(TMP_CONFIG),
                nodes=nodes, tasks=ntasks_per_node, mem=mem, plane=plane,
                exclude=exclude, mpi_wrapper=mpi_wrapper,
                container=containers[i], partition=partition, time=time,
                name=jobname, runname=name,
                SPWs=crosscal_kwargs['spw'], nspw=crosscal_kwargs['nspw'],
                account=account, reservation=reservation,
                modules=modules, justrun=justrun,
            )
        else:
            write_sbatch(
                script, '--config {0}'.format(TMP_CONFIG),
                nodes=1, tasks=1, mem=mem, plane=1,
                mpi_wrapper='srun', container=containers[i],
                partition=partition, time=time, name=jobname, runname=name,
                SPWs=crosscal_kwargs['spw'], nspw=crosscal_kwargs['nspw'],
                exclude=exclude, account=account, reservation=reservation,
                modules=modules, justrun=justrun,
            )

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
