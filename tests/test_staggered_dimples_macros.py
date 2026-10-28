"""Static compatibility checks for the APDL inputs (not MAPDL execution)."""
from pathlib import Path
import math
import re

ROOT=Path(__file__).resolve().parents[1]
APDL=ROOT/'apdl'/'standalone'


def test_staggered_dimples_macros():
    unit=(APDL/'03_RUN_staggered_dimples_UNIT.mac').read_text()
    phased=(APDL/'04_RUN_staggered_dimples_PHASED.mac').read_text()
    for s in (unit,phased):
        assert s.count('VSBV,TUBEID,TOOLID') == 2
        assert s.count('SPHERE,0,NOTCH') == 2
        assert 'NOTCH1_Z=-3.0' in s and 'NOTCH2_Z=3.0' in s
        assert 'CMSEL,S,GAUGEBOTH' in s and 'CMSEL,A,GAUGEB' in s
        assert 'SET,LAST' in s and 'ABS(RES_TIME-IC)' in s
        assert 'TIME,IC' in s and 'SOLVE' in s
        assert 'CLOSED_END=1' in s
        assert all(x in s for x in ('fe_stress,csv','fe_strain,csv',
            'fe_coordinates,csv','fe_result_checks,csv','fe_gauge_summary,csv'))
        assert not re.search(r'(?mi)^\s*(\*USE|/INPUT),',s)
        assert max(map(len,s.splitlines())) <= 128
    assert '\nBASIS_MODE=1\n' in unit
    assert '\nBASIS_MODE=0\n' in phased
    for i,j,v in [(1,1,'18'),(2,1,'8.48528137423857'),
                   (3,1,'8.48528137423857'),(1,2,'500'),(2,2,'5000'),
                   (1,3,'1500'),(3,3,'26000')]:
        assert f'CTRL({i},{j})={v}' in phased
    assert math.isclose(math.hypot(8.48528137423857,8.48528137423857),12)
