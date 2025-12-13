import numpy as np

import tangos
from tangos.util import consistent_collection

from ... import core, temporary_halolist
from ...core import extraction_patterns
from .. import (
    BuiltinFunction,
    Calculation,
    FixedInput,
    FixedNumericInput,
    LiveProperty,
    StoredProperty,
)


@BuiltinFunction.register
def match(source_halos, target):
    if target is None:
        results = [None]*len(source_halos)
    else:
        from ... import relation_finding
        if not isinstance(target, core.Base):
            target = tangos.get_item(target, core.Session.object_session(source_halos[0]))
        results = relation_finding.MultiSourceMultiHopStrategy(source_halos, target).all()
    # if following assert fails, it might be duplicate links in the database which the
    # current MultiSourceMultiHop implementation cannot de-duplicate:
    assert len(results) == len(source_halos)
    return np.array(results, dtype=object)
match.set_input_options(0, provide_proxy=True, assert_class = FixedInput)


@BuiltinFunction.register
def match_reduce(source_halos: list[core.halo.Halo],
                 target_name: str,
                 calculation: LiveProperty,
                 reduction: str):
    """Get the reduction (sum, mean, min, max) of the specified calculation over all objects linked in the
    specified target timestep or simulation"""
    if len(source_halos) == 0:
        return []

    calculation = calculation.name

    if not isinstance(calculation, Calculation):
        calculation = tangos.live_calculation.parser.parse_property_name(calculation)

    reduction_map = {'sum': np.sum,
                     'mean': np.mean,
                     'min': lambda input: np.min(input) if len(input)>0 else None,
                     'max': lambda input: np.max(input) if len(input)>0 else None}
    if reduction not in reduction_map.keys():
        raise ValueError(f"Unsupported reduction '{reduction}' in match_reduce. Supported reductions are sum, mean, min, max.")

    from ... import relation_finding

    target = tangos.get_item(target_name, core.Session.object_session(source_halos[0]))
    strategy = relation_finding.MultiSourceMultiHopStrategy(source_halos, target, one_match_per_input=False)

    # using strategy.temp_table doesn't seem to offer access to the sources of the halos, so we
    # take a pass through python. This also offers the opportunity to use a throw-away session
    # for the onwards calculation. There may be more efficient ways to do all this.

    all_halos = strategy.all()
    all_sources = strategy.sources()

    with core.Session() as session:
        with temporary_halolist.temporary_halolist_table(session, [h.id for h in all_halos]) as tt:
            target_halos_supplemented = calculation.supplement_halo_query(
                temporary_halolist.halo_query(tt)
            )
            values,  = calculation.values(target_halos_supplemented.all())

    values_per_halo = [[] for _ in source_halos]
    for source, value in zip(all_sources, values):
        values_per_halo[source].append(value)

    reduction_func = reduction_map[reduction]
    return [reduction_func(vals) for vals in values_per_halo]



match_reduce.set_input_options(2, assert_class=FixedInput, provide_proxy=True)
match_reduce.set_input_options(0, assert_class=FixedInput, provide_proxy=True)
match_reduce.set_input_options(1, assert_class=Calculation, provide_proxy=True)


@BuiltinFunction.register
def later(source_halos, num_steps):
    timestep = consistent_collection.ConsistentCollection(source_halos).timestep.get_next(num_steps)
    return match(source_halos, timestep)

later.set_input_options(0, provide_proxy=True, assert_class = FixedNumericInput)


@BuiltinFunction.register
def earlier(source_halos, num_steps):
    return later(source_halos, -num_steps)

earlier.set_input_options(0, provide_proxy=True, assert_class = FixedNumericInput)


@BuiltinFunction.register
def latest(source_halos):
    from .search import find_descendant
    return find_descendant(source_halos, LiveProperty('t').proxy_value(), 'max')


@BuiltinFunction.register
def earliest(source_halos):
    from .search import find_progenitor
    return find_progenitor(source_halos, LiveProperty('t').proxy_value(), 'min')

@BuiltinFunction.register
def has_property(source_halos, property):
    from ...util import is_not_none
    return is_not_none(property)

has_property.set_input_options(0, provide_proxy=False, assert_class=StoredProperty)

@has_property.set_initialisation
def has_property_init(input):
    input.set_extraction_pattern(extraction_patterns.HaloPropertyRawValueGetter())


from . import arithmetic, array, link, reassembly, search
