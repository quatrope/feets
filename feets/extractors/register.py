import inspect

from .extractor import (
    DATAS,
    Extractor,
    ExtractorBadDefinedError,
)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class FeatureNotFound(ValueError):
    pass


class FeatureExtractorAlreadyRegistered(ValueError):
    pass


# =============================================================================
# REGISTER UTILITY
# =============================================================================

_extractors = {}


def is_instance_or_is_extractor(obj):
    return isinstance(obj, Extractor) or (
        inspect.isclass(obj) and issubclass(obj, Extractor)
    )


def register_extractor(cls):
    if not issubclass(cls, Extractor):
        msg = f"'cls' must be a subclass of Extractor. Found: {cls}"
        raise TypeError(msg)
    for dependency in cls.get_dependencies():
        if dependency not in _extractors.keys():
            msg = f"Dependency '{dependency}' from extractor '{cls}'"
            raise ExtractorBadDefinedError(msg)
    for feature in cls.get_features():
        if is_feature_registered(feature):
            raise FeatureExtractorAlreadyRegistered(feature)
        _extractors[feature] = cls
    return cls


def unregister_extractor(cls):
    pass


def registered_extractors():
    return dict(_extractors)


def is_feature_registered(feature):
    return feature in _extractors


def is_extractor_registered(cls):
    if not issubclass(cls, Extractor):
        msg = f"'cls' must be a subclass of Extractor. Found: '{cls}'"
        raise TypeError(msg)
    return cls in _extractors.values()


def available_features():
    return sorted(_extractors.keys())


def extractor_of(feature):
    if not is_feature_registered(feature):
        raise FeatureNotFound(feature)
    return _extractors[feature]


def sort_by_dependencies(exts, retry=None):
    """Calculate the Feature Extractor Resolution Order."""
    sorted_ext, features_from_sorted = [], set()
    pending = [(e, 0) for e in exts]
    retry = len(exts) * 100 if retry is None else retry
    while pending:
        ext, cnt = pending.pop(0)

        if not isinstance(ext, Extractor) and not issubclass(ext, Extractor):
            msg = (
                f"Only Extractor instances are allowed. Found: '{type(ext)}'."
            )
            raise TypeError(msg)

        deps = ext.get_dependencies()
        if deps.difference(features_from_sorted):
            if cnt + 1 > retry:
                msg = (
                    f"Maximum retry ({retry}) to sort achieved by "
                    f"extractor '{type(ext)}'."
                )
                raise RuntimeError(msg)
            pending.append((ext, cnt + 1))
        else:
            sorted_ext.append(ext)
            features_from_sorted.update(ext.get_features())
    return tuple(sorted_ext)


def extractors_from_data(data):
    diff = set(data).difference(DATAS)
    if diff:
        msg = f"Invalid data(s): {', '.join(diff)}."
        raise ValueError(msg)

    extractors = set()
    for extractor in _extractors.values():
        if extractor.get_data().issubset(data):
            extractors.add(extractor)
    return extractors


def extractors_from_features(features):
    extractors = set()
    for feature in features:
        if not is_feature_registered(feature):
            raise FeatureNotFound(feature)
        extractors.add(extractor_of(feature))
    return set(extractors)


def get_execution_plan(*, data=None, only=None, exclude=None):
    if not set(only or []).isdisjoint(exclude or []):
        raise ValueError("Features in 'only' and 'exclude' must be disjoint.")

    from_data = (
        extractors_from_data(data)
        if data is not None
        else _extractors.values()
    )
    from_only = (
        extractors_from_features(only)
        if only is not None
        else _extractors.values()
    )
    from_exclude = (
        extractors_from_features(exclude) if exclude is not None else set([])
    )

    selected_extractors = (
        set(from_data).intersection(from_only).difference(from_exclude)
    )

    return sort_by_dependencies(selected_extractors)
