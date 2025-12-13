import numpy as np
from pytest import raises as assert_raises

import tangos as db
import tangos.testing as testing
import tangos.testing.simulation_generator


def setup_module():
    testing.init_blank_db_for_testing()

    generator = tangos.testing.simulation_generator.SimulationGeneratorForTests()
    generator.add_timestep()
    ts1_h1, ts1_h2, ts1_h3, ts1_h4, ts1_h5 = generator.add_objects_to_timestep(5)
    generator.add_timestep()
    ts2_h1, ts2_h2, ts2_h3 = generator.add_objects_to_timestep(3)


    generator.link_last_halos_using_mapping({1: 1, 2: 3, 3: 2, 4: 3, 5: 2})

    generator.add_timestep()
    ts3_h1, = generator.add_objects_to_timestep(1)
    generator.link_last_halos()

    ts1_h1['val'] = 1.0
    ts1_h2['val'] = 2.0
    ts1_h3['val'] = 3.0
    ts1_h4['val'] = 4.0
    ts1_h5['val'] = 5.0

    ts2_h1['val'] = 10.0
    ts2_h2['val'] = 20.0
    ts2_h3['val'] = 30.0

    ts3_h1['val'] = 100.0

    db.core.get_default_session().commit()

def teardown_module():
    tangos.core.close_db()

def test_reduce_function():
    results, = db.get_timestep('sim/ts2').calculate_all('match_reduce("sim/ts1", val * 2.0, "min")')
    assert results[0] == 2.0
    assert results[1] == 6.0
    assert results[2] == 4.0

def test_reduce_min():
    results, = db.get_timestep('sim/ts2').calculate_all('match_reduce("sim/ts1", val, "min")')
    assert results[0] == 1.0
    assert results[1] == 3.0
    assert results[2] == 2.0

def test_reduce_max():
    results, = db.get_timestep('sim/ts2').calculate_all('match_reduce("sim/ts1", val, "max")')
    assert results[0] == 1.0
    assert results[1] == 5.0
    assert results[2] == 4.0

def test_reduce_mean():
    results, = db.get_timestep('sim/ts2').calculate_all('match_reduce("sim/ts1", val, "mean")')
    assert results[0] == 1.0
    assert results[1] == 4.0
    assert results[2] == 3.0

def test_reduce_sum():
    results, = db.get_timestep('sim/ts2').calculate_all('match_reduce("sim/ts1", val, "sum")')
    assert results[0] == 1.0
    assert results[1] == 8.0
    assert results[2] == 6.0

def test_unsupported_reduce():
    with assert_raises(ValueError):
        db.get_timestep('sim/ts2').calculate_all('match_reduce("sim/ts1", val, "unsupported")')

def test_reduce_no_matches():
    hnum, results, = db.get_timestep('sim/ts2').calculate_all('halo_number()',
                                                        'match_reduce("sim/ts3", val, "max")')
    assert results == [100.0]
    assert hnum == [1]

    hnum, results, = db.get_timestep('sim/ts2').calculate_all('halo_number()',
                                                              'match_reduce("sim/ts3", val, "sum")')
    assert np.allclose(results, [100.0, 0.0, 0.0])
    assert (hnum == [1, 2, 3]).all()
