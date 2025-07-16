import os
from stack_plan import generate_stacking_plan


def test_generate_stacking_plan_basic(tmp_path):
    results = [
        {
            'status': 'ok',
            'mount': 'EQ',
            'bortle': '3',
            'telescope': 'C11',
            'date_obs': '2024-07-13T22:00:00',
            'filter': 'R',
            'exposure': 180,
            'path': '/astro/C11/2024-07-13/img1.fit',
        },
        {
            'status': 'ok',
            'mount': 'EQ',
            'bortle': '3',
            'telescope': 'C11',
            'date_obs': '2024-07-13T22:10:00',
            'filter': 'R',
            'exposure': 180,
            'path': '/astro/C11/2024-07-13/img2.fit',
        },
        {
            'status': 'ok',
            'mount': 'ALTZ',
            'bortle': '5',
            'telescope': 'Seestar',
            'date_obs': '2024-07-12T20:00:00',
            'filter': 'N/A',
            'exposure': 10,
            'path': '/astro/Seestar/2024-07-12/img99.fit',
        },
    ]
    plan = generate_stacking_plan(
        results,
        include_exposure_in_batch=False,
        criteria={},
        sort_spec=[('telescope', False), ('session_date', False)],
    )
    assert len(plan) == 3
    assert plan[0]['batch_id'] == 'C11_2024-07-13_R'
    assert plan[2]['batch_id'] == 'Seestar_2024-07-12_N/A'

