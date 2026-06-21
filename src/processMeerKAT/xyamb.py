#!/usr/bin/env python
#
# xyamb.py (local, trimmed)
#
# Originally from CASA's recipes.almapolhelpers (gmoellen, 2013/2015).
# CASA 6 removed this module, but we only need xyamb() for X-Y phase
# ambiguity resolution in xy_yx_solve.py. Ported to Python 3 / casatools.
#
# Only xyamb() is retained and maintained here.

import os
from math import pi, atan2

import numpy as np
from casatools import table

import logging
logger = logging.getLogger(__name__)


def xyamb(xytab, qu, xyout=''):
    """Resolve the X-Y phase ambiguity in a cross-hand phase cal table.

    Parameters
    ----------
    xytab : str
        Input X-Y phase cal table.
    qu : tuple
        (Q, U) expected fractional polarization.
    xyout : str
        Output table; defaults to xytab (modified in place).

    Returns
    -------
    list
        Stokes vector [I, Q, U, V] with the resolved Q, U.
    """
    if not isinstance(qu, tuple):
        raise TypeError('qu must be a tuple: (Q,U)')

    if xyout == '':
        xyout = xytab
    if xyout != xytab:
        os.system('cp -r ' + xytab + ' ' + xyout)

    QUexp = complex(qu[0], qu[1])
    logger.info('Expected QU = {0}'.format(qu))

    mytb = table()
    mytb.open(xyout, nomodify=False)

    QU = mytb.getkeyword('QU')['QU']
    P = np.sqrt(QU[0, :]**2 + QU[1, :]**2)

    nspw = P.shape[0]
    for ispw in range(nspw):
        st = mytb.query('SPECTRAL_WINDOW_ID==' + str(ispw))
        if st.nrows() > 0:
            q = QU[0, ispw]
            u = QU[1, ispw]
            qufound = complex(q, u)
            c = st.getcol('CPARAM')
            fl = st.getcol('FLAG')
            xyph0 = np.angle(np.mean(c[0, :, :][np.logical_not(fl[0, :, :])]), deg=True)
            logger.info('Spw = {0}: Found QU = {1}'.format(ispw, QU[:, ispw]))
            if np.absolute(np.angle(qufound / QUexp) * 180 / pi) > 90.0:
                c[0, :, :] *= -1.0
                xyph1 = np.angle(np.mean(c[0, :, :][np.logical_not(fl[0, :, :])]), deg=True)
                st.putcol('CPARAM', c)
                QU[:, ispw] *= -1
                logger.info('   ...CONVERTING X-Y phase from {0} to {1} deg'.format(xyph0, xyph1))
            else:
                logger.info('      ...KEEPING X-Y phase {0} deg'.format(xyph0))
            st.close()

    QUr = {'QU': QU}
    mytb.putkeyword('QU', QUr)
    mytb.close()

    QUm = np.mean(QU[:, P > 0], 1)
    QUe = np.std(QU[:, P > 0], 1)
    Pm = np.sqrt(QUm[0]**2 + QUm[1]**2)
    Xm = 0.5 * atan2(QUm[1], QUm[0]) * 180 / pi

    logger.info('Ambiguity resolved (spw mean): Q={0} U={1} (rms={2} {3}) P={4} X={5}'.format(
        QUm[0], QUm[1], QUe[0], QUe[1], Pm, Xm))

    stokes = [1.0, QUm[0], QUm[1], 0.0]
    logger.info('Returning the following Stokes vector: {0}'.format(stokes))

    return stokes
