#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

#!/usr/bin/env python3
import sys
import os
import numpy as np

from . import processMeerKAT
from . import config_parser

from casatasks import *
from casatools import msmetadata,table,measures,quanta

logger = processMeerKAT.logger

# Get access to the msmd module
msmd = msmetadata()
tb = table()
me = measures()
qa = quanta()

def get_fields(MS):

    """Extract field numbers from intent, including calibrators for bandpass, flux, phase & amplitude, and the target. Only the
    target allows for multiple field IDs, while all others extract the field with the most scans and put all other IDs as target fields.

    Arguments:
    ----------
    MS : str
        Input MeasurementSet (relative or absolute path).

    Returns:
    --------
    fieldIDs : dict
        fluxfield : int
            Field for total flux calibration.
        bpassfield : int
            Field for bandpass calibration.
        phasecalfield : int
            Field for phase calibration.
        targetfields : int
            Target field."""

    fieldIDs = {}
    extra_fields = []
    intents = msmd.intents()

    #Set default for any missing intent as field for intent CALIBRATE_FLUX
    fluxcal = msmd.fieldsforintent('CALIBRATE_FLUX')
    if 'CALIBRATE_FLUX' not in intents or fluxcal.size == 0:
        logger.error('You must have a field with intent "CALIBRATE_FLUX". I only found {0} in dataset "{1}".'.format(intents,MS))
        return fieldIDs
    else:
        default = fluxcal[0]

    #Use 'CALIBRATE_PHASE' or if missing, 'CALIBRATE_AMPLI'
    phasecal_intent = 'CALIBRATE_PHASE'
    if phasecal_intent not in intents:
        phasecal_intent = 'CALIBRATE_AMPLI'

    fieldIDs['fluxfield'] = get_field(MS,'CALIBRATE_FLUX','fluxfield',extra_fields)
    fieldIDs['bpassfield'] = get_field(MS,'CALIBRATE_BANDPASS','bpassfield',extra_fields,default=default)
    fieldIDs['phasecalfield'] = get_field(MS,phasecal_intent,'phasecalfield',extra_fields,default=default)
    fieldIDs['targetfields'] = get_field(MS,'TARGET','targetfields',extra_fields,default=default,multiple=True)

    if 'UNKNOWN' in intents:
        err_msg = "Intent 'UNKNOWN' present in MS but couldn't find any fields with this intent. Please append any extra field manually."
        if len(msmd.fieldsforintent('UNKNOWN')) > 0:
            try:
                polfields = np.array(msmd.namesforfields(msmd.fieldsforintent('UNKNOWN'))) #bogus MeerKAT mislabelling during conversion to MS
                for polfield in polfields:
                    if polfield not in extra_fields:
                        extra_fields.append(polfield)
            except RuntimeError as e:
                logger.warning(err_msg)
        else:
            logger.warning(err_msg)


    #Put any extra fields in extra_fields
    if len(extra_fields) > 0:
        fieldIDs['extrafields'] = ','.join([str(extra_fields[i]) for i in range(len(extra_fields))])

    return fieldIDs


def get_field(MS,intent,fieldname,extra_fields,default=0,multiple=False):

    """Extract field IDs based on intent. When multiple fields are present, if multiple is True, return a
    comma-seperated string, otherwise return a single field string corresponding to the field with the most scans.

    Arguments:
    ----------
    MS : str
        Input MeasurementSet (relative or absolute path).
    intent : str
        Calibration intent.
    fieldname : str
        The name given by the pipeline to the field being extracted (for output).
    extra_fields : list
        List of extra fields (passed by reference).
    default : int, optional
        Default field to return if intent missing.
    multiple : bool, optional
        Allow multiple fields?

    Returns:
    --------
    fieldIDs : str
        Extracted field ID(s), comma-seperated for multiple fields."""

    fields = np.array(msmd.namesforfields(msmd.fieldsforintent(intent)))

    if fields.size == 0:
        logger.warning('Intent "{0}" not found in dataset "{1}". Setting to "{2}"'.format(intent,MS,default))
        fieldIDs = str(default)
    elif fields.size == 1:
        fieldIDs = str(fields[0])
    else:
        logger.info('Multiple fields found with intent "{0}" in dataset "{1}" - {2}.'.format(intent,MS,fields))

        if multiple:
            logger.info('Will use all of them for "{0}".'.format(fieldname))
            fieldIDs = ','.join([str(fields[i]) for i in range(fields.size)])
        else:
            maxfield, maxscan = 0, 0
            scans = [msmd.scansforfield(ff) for ff in fields]
            # scans is an array of arrays
            for ind, ss in enumerate(scans):
                if len(ss) > maxscan:
                    maxscan = len(ss)
                    maxfield = fields[ind]

            logger.warning('Only using field "{0}" for "{1}", which has the most scans ({2}).'.format(maxfield,fieldname,maxscan))
            fieldIDs = str(maxfield)

            #Put any extra fields with the same intent in extra fields
            extras = list(set(fields) - set(extra_fields) - set([maxfield]))
            if len(extras) > 0:
               logger.warning('Putting extra fields with intent "{0}" in "extrafields" - {1}'.format(intent,extras))
               extra_fields += extras

    return fieldIDs

def check_refant(MS,refant,config,warn=True):

    """Check if reference antenna exists, otherwise throw an error or display a warning.

    Arguments:
    ----------
    MS : str
        Input MeasurementSet (relative or absolute path).
    refant: str
        Input reference antenna.
    config : str
        Path to config file.
    warn : bool, optional
        Warn the user? If False, raise ValueError."""

    try:
        refant = int(refant)
    except ValueError: # It's not an int, but a str
        pass

    msmd.open(MS)
    if type(refant) is str:
        ants = msmd.antennanames()
    else:
        ants = msmd.antennaids()

    if refant not in ants:
        err = "Reference antenna '{0}' isn't present in input dataset '{1}'. Antennas present are: {2}. Try 'm052' or 'm005' if present, or ensure 'calcrefant=True' and 'calc_refant.py' script present in '{3}'.".format(refant,MS,ants,config)
        if warn:
            logger.warning(err)
        else:
            raise ValueError(err)
    else:
        logger.info("Using reference antenna '{0}'.".format(refant))
        if refant == 'm059':
            logger.info("This is usually a well-behaved (stable) antenna. Edit '{0}' to change this, by updating 'refant' in [crosscal] section.".format(config))
            logger.debug("Alternatively, set 'calcrefant=True' in [crosscal] section of '{0}', and include 'calc_refant.py' in 'scripts' in [slurm] section.".format(config)) #(included by default)

def auto_detect_spw(msmd):

    """Build the SPW string and count from the MS's native SPECTRAL_WINDOW structure.

    Honors whatever SPW boundaries the input MS already has rather than imposing
    a linspace split across a user-supplied frequency range. Each detected SPW
    is emitted as ``<id>:<low>~<high>MHz`` so that downstream ``mstransform``
    selects the exact native SPW (not a freq-overlap from any SPW).

    Arguments:
    ----------
    msmd : casatools.msmetadata
        An already-opened msmetadata handle on the input MS.

    Returns:
    --------
    spw_string : str
        Comma-separated SPW selection string, e.g. ``"0:880~933MHz,1:960~1010MHz"``.
    nspw : int
        Number of detected SPWs (== msmd.nspw())."""

    nspw = msmd.nspw()
    parts = []
    for i in range(nspw):
        freqs_hz = msmd.chanfreqs(i)
        low_mhz = float(freqs_hz[0]) / 1e6
        high_mhz = float(freqs_hz[-1]) / 1e6
        if low_mhz > high_mhz:
            low_mhz, high_mhz = high_mhz, low_mhz
        parts.append('{0}:{1:.3f}~{2:.3f}MHz'.format(i, low_mhz, high_mhz))
    return ','.join(parts), nspw

def resolve_spw_for_build(msmd, requested_nspw):

    """Decide what SPW string and nspw to write to the config during -B.

    Policy:
      - If the MS has multiple native SPWs, honor that structure exactly: write
        one entry per native SPW and force ``nspw == msmd.nspw()``. The user's
        requested_nspw is ignored (with a warning if it differed).
      - If the MS has a single native SPW, expose only its frequency bounds in
        the SPW string and preserve the user's requested_nspw so that
        downstream ``spw_split`` can subdivide it for parallelism. This is the
        common L-band case: one wide SPW that we want to break into N chunks.

    Arguments:
    ----------
    msmd : casatools.msmetadata
        Open msmetadata handle.
    requested_nspw : int
        The nspw value already in the config (typically the user's choice).

    Returns:
    --------
    spw_string : str
        Comma-separated SPW selection, e.g. ``"0:880~933MHz,1:960~1010MHz"`` or
        ``"0:880~1680MHz"``.
    nspw : int
        The nspw value to write to ``[crosscal] nspw``."""

    native_spw, native_nspw = auto_detect_spw(msmd)

    if native_nspw > 1:
        if requested_nspw and requested_nspw != native_nspw:
            logger.warning(
                "Input MS has {0} native SPWs; overriding configured nspw={1} to match.".format(
                    native_nspw, requested_nspw
                )
            )
        return native_spw, native_nspw

    # Single native SPW: keep the user's nspw so spw_split can subdivide.
    if requested_nspw and requested_nspw > 1:
        logger.info(
            "Input MS has a single native SPW; will subdivide into {0} chunks "
            "for parallelism (set nspw=1 in config to disable).".format(requested_nspw)
        )
    return native_spw, requested_nspw or 1

def check_spw(config,msmd):

    """Check SPW bounds are within the SPW bounds of the MS. If not, output a warning and update the SPW.

    Arguments:
    ----------
    config : str
        Path to config file.

    Returns:
    --------
    The SPW to be written to the config, potentially udpated."""

    update = False
    low,high,unit,dirs = config_parser.parse_spw(config)
    lowest = np.min(low)
    highest = np.max(high)
    nspw = msmd.nspw()

    # if nspw > 1:
    #     logger.warning("Expected 1 SPW but found nspw={0}. Please manually edit 'spw' in '{1}'.".format(nspw,config))

    #Check SPW bounds overlap with MS, and convert to MHz explicitly, assuming first and last SPW contain lowest and highest frequencies
    ms_low = msmd.chanfreqs(0)[0] / 1e6
    ms_high = msmd.chanfreqs(nspw-1)[-1] / 1e6

    if type(unit) is list:
        low_unit = unit[low.index(lowest)]
        high_unit = unit[high.index(highest)]
    else:
        low_unit = unit
        high_unit = unit

    #i.e. channel number
    if low_unit == '':
        low_MHz = msmd.chanfreqs(0)[low] / 1e6
    if high_unit == '':
        high_MHz = msmd.chanfreqs(nspw-1)[high] / 1e6
    else:
        low_MHz=qa.convertfreq('{0}{1}'.format(lowest,low_unit),'MHz')['value']
        high_MHz=qa.convertfreq('{0}{1}'.format(highest,high_unit),'MHz')['value']

    if low_MHz < ms_low - 1:
        low_MHz = int(ms_low)
        update = True
    if high_MHz > ms_high + 1:
        high_MHz = int(round(ms_high+0.5))
        update = True

    SPW = '*:{0}~{1}MHz'.format(low_MHz,high_MHz)

    if update:
        logger.warning('Default SPW outside SPW of input MS ({0}~{1}MHz). Forcing SPW={2}'.format(ms_low,ms_high,SPW))

    return SPW

def parang_coverage(vis, calfield):

    """Check whether the parallactic angle coverage of the phase calibrator field is > 30 degrees, necessary to do polarisation calibration.

    Arguments:
    ----------
    vis : str
        Input MeasurementSet (relative or absolute path).
    calfield : int
        Phase calibrator field ID.

    Returns:
    --------
    delta_parang : float
        The parallactic angle coverage of the phase calibrator field."""

    tb.open(vis+'::ANTENNA')
    pos = tb.getcol('POSITION')
    meanpos = np.mean(pos, axis=1)
    frame = tb.getcolkeyword('POSITION','MEASINFO')['Ref']
    units = tb.getcolkeyword('POSITION','QuantumUnits')
    mpos  = me.position(frame,
                    str(meanpos[0])+units[0],
                    str(meanpos[1])+units[1],
                    str(meanpos[2])+units[2])
    me.doframe(mpos)
    tb.close()

    # _geodetic_ latitude
    latr=me.measure(mpos,'WGS84')['m1']['value']
    tb.open(vis+'::FIELD')
    srcid = tb.getcol('SOURCE_ID')
    dirs=tb.getcol('DELAY_DIR')[:,0,:]
    tb.close()
    tb.open(vis,nomodify=True)
    st = tb.query('FIELD_ID=='+str(calfield))

    # get time stamps of first and last row
    nrows = st.nrows()
    tbeg = st.getcol('TIME', startrow=0, nrow=1)[0]
    tend = st.getcol('TIME', startrow=nrows-1, nrow=1)[0]

    # calculate parallactic angles for first and last time
    parang = np.zeros(2)

    # calculate parallactic angle
    rah = dirs[0,calfield]*12.0/np.pi
    decr = dirs[1,calfield]

    for itim, ts in enumerate([tbeg, tend]):
        tm = me.epoch('UTC',str(ts)+'s')
        last = me.measure(tm,'LAST')['m0']['value']
        last -= np.floor(last)  # days
        last *= 24.0  # hours
        ha = last-rah  # hours
        har = ha*2.0*np.pi/24.0
        parang[itim] = np.arctan2((np.cos(latr)*np.sin(har)), (np.sin(latr)*np.cos(decr) - np.cos(latr)*np.sin(decr)*np.cos(har)))

    delta_parang = np.rad2deg(parang[1] - parang[0])
    logger.debug("Delta parang: {0}".format(delta_parang))
    tb.close()

    return np.abs(delta_parang)


def get_xy_field(visname, fields):
    """
    From the input MS determine which field should
    be used for XY-phase calibration (if required).

    In the following order :
    3C286
    3C138
    secondaryfield (nominally dpolfield)
    """

    msmd.open(visname)
    fieldnames = msmd.fieldnames()
    msmd.done()

    # Use 3C286 or 3C138 if present in the data
    calibrator_3C286 = set(["3C286", "1328+307", "1331+305", "J1331+3030"]).intersection(set(fieldnames))
    calibrator_3C138 = set(["3C138", "0518+165", "0521+166", "J0521+1638"]).intersection(set(fieldnames))

    if calibrator_3C286:
        xyfield = list(calibrator_3C286)[0]
    elif calibrator_3C138:
        xyfield = list(calibrator_3C138)[0]
    else:
        xyfield = fields.dpolfield

    return xyfield

def main():

    args = processMeerKAT.parse_args()
    processMeerKAT.setup_logger(args.config,args.verbose)
    msmd.open(args.MS)

    dopol = args.dopol
    refant = config_parser.parse_config(args.config)[0]['crosscal']['refant']
    fields = get_fields(args.MS)
    logger.info('[fields] section written to "{0}". Edit this section if you need to change field IDs (comma-seperated string for multiple IDs, not supported for calibrators).'.format(args.config))

    npol = msmd.ncorrforpol()[0]
    parang = 0
    if 'phasecalfield' in fields:
        calfield = msmd.fieldsforname(fields['phasecalfield'])[0]
        parang = parang_coverage(args.MS, calfield)

    if npol < 4:
        logger.warning("Only {0} polarisations present in '{1}'. Any attempted polarisation calibration will fail, so setting dopol=False in [run] section of '{2}'.".format(npol,args.MS,args.config))
        dopol = False
    elif 0 < parang < 30:
        logger.warning("Parallactic angle coverage is < 30 deg. Polarisation calibration will most likely fail, so setting dopol=False in [run] section of '{0}'.".format(args.config))
        dopol = False

    check_refant(args.MS, refant, args.config, warn=True)

    existing_nspw = config_parser.parse_config(args.config)[0].get('crosscal', {}).get('nspw', 1)
    try:
        existing_nspw = int(existing_nspw)
    except (TypeError, ValueError):
        existing_nspw = 1

    SPW, write_nspw = resolve_spw_for_build(msmd, existing_nspw)

    config_parser.overwrite_config(args.config, conf_dict={'dopol' : dopol}, conf_sec='state', sec_comment='# Pipeline runtime state — do not edit manually')
    config_parser.overwrite_config(args.config, conf_dict=fields, conf_sec='fields')
    config_parser.overwrite_config(args.config, conf_dict={'spw': SPW, 'nspw': write_nspw}, conf_sec='crosscal')

    msmd.done()

if __name__ == "__main__":
    main()
