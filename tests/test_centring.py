import numpy as np
import pytest

pynbody = pytest.importorskip("pynbody")

from tangos.properties.pynbody.centring import CentreAndRadius


def _make_halo(true_center):
    """Build an in-memory pynbody snapshot containing a single compact halo whose true centre is
    `true_center`, wrapped into the periodic box [-50, 50]."""
    pos = np.asarray(true_center, dtype=float) + np.random.normal(scale=3, size=(2000, 3))
    # wrap into [-L/2, L/2), matching pynbody's 'center' convention:
    pos = (pos + 50.0) % 100.0 - 50.0

    f = pynbody.new(dm=2000)
    f['pos'] = pos
    f['pos'].units = 'kpc'
    f['mass'] = np.ones(2000)
    f['mass'].units = 'Msol'
    f.properties['boxsize'] = pynbody.units.Unit('%f kpc' % 100.0)
    return f


def _run(halo):
    calc = CentreAndRadius(None)
    with calc.timing_monitor(calc):  # so mark_timer() has somewhere to record
        return calc._get_centre_and_max_radius(halo.dm)


def test_centre_in_box_for_edge_halo():
    """A halo sitting on the periodic boundary must still yield a centre inside the box."""
    true_center = np.array([50.0, 0.0, 0.0])  # exactly on the x-edge (== -L/2)
    f = _make_halo(true_center)

    center, rmax = _run(f)

    assert np.all(center >= -50.0)
    assert np.all(center < 50.0)

    # centre should match the true centre up to periodic wrapping and shrink-sphere noise:
    delta = (center - true_center + 50.0) % 100.0 - 50.0
    assert np.linalg.norm(delta) < 1.0


def test_max_radius_matches_minimum_image():
    """max_radius must equal the true minimum-image radius even for an edge-straddling halo."""
    true_center = np.array([50.0, 0.0, 0.0])
    f = _make_halo(true_center)

    center, rmax = _run(f)

    d = np.asarray(f.dm['pos']) - center
    d -= 100 * np.round(d / 100)  # minimum image
    r_expected = np.sqrt((d ** 2).sum(1)).max()
    assert rmax == pytest.approx(r_expected, rel=1e-6)


def test_interior_halo_unchanged():
    """A halo comfortably inside the box should be centred correctly too (regression guard)."""
    true_center = np.array([10.0, -5.0, 20.0])
    f = _make_halo(true_center)

    center, rmax = _run(f)

    delta = center - true_center
    assert np.linalg.norm(delta) < 1.0
    assert np.all(center >= -50.0)
    assert np.all(center < 50.0)
