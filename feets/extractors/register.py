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
    for d in cls.get_dependencies():
        if d not in _extractors.keys():
            msg = f"Dependency '{d}' from extractor '{cls}'"
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


def get_extractors_by_data(data=None):
    if data is None:
        return set(registered_extractors().values())

    diff = set(data).difference(DATAS)
    if diff:
        msg = f"Invalid data(s): {', '.join(diff)}."
        raise ValueError(msg)

    extractors = set()
    for extractor_cls in registered_extractors().values():
        if extractor_cls.get_data().intersection(data):
            extractors.add(extractor_cls)

    return extractors


def get_extractors(features=None):
    selected_features = available_features() if features is None else features

    extractors = set()
    for feature in selected_features:
        if not is_feature_registered(feature):
            raise FeatureNotFound(feature)
        extractors.add(extractor_of(feature))

    return extractors


def get_plan(*, data=None, only=None, exclude=None):
    if set(only or []).intersection(exclude or []):
        msg = "Features in 'only' and 'exclude' must be disjoint."
        raise ValueError(msg)

    from_data = get_extractors_by_data(data=data)
    from_only = get_extractors(features=only)
    from_exclude = get_extractors(features=exclude)

    selected_extractors = from_data.intersection(from_only).difference(
        from_exclude
    )

    return sort_by_dependencies(selected_extractors)
