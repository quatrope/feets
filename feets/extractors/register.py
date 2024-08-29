import inspect

from . import core

# =============================================================================
# REGISTER UTILITY
# =============================================================================

_extractors = {}


def is_isntance_or_is_extractor(obj):
    return isinstance(obj, core.Extractor) or (
        inspect.isclass(obj) and issubclass(obj, core.Extractor)
    )


def register_extractor(cls):
    if not issubclass(cls, core.Extractor):
        msg = "'cls' must be a subclass of Extractor. Found: {}"
        raise TypeError(msg.format(cls))
    for d in cls.get_dependencies():
        if d not in _extractors.keys():
            msg = "Dependency '{}' from extractor {}".format(d, cls)
            raise core.ExtractorBadDefinedError(msg)
    _extractors.update((f, cls) for f in cls.get_features())
    return cls


def registered_extractors():
    return dict(_extractors)


def is_feature_registered(feature):
    return feature in _extractors


def is_extractor_registered(cls):
    if not issubclass(cls, core.Extractor):
        msg = "'cls' must be a subclass of Extractor. Found: {}"
        raise TypeError(msg.format(cls))
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

        if not isinstance(ext, core.Extractor) and not issubclass(
            ext, core.Extractor
        ):
            msg = "Only Extractor instances are allowed. Found {}."
            raise TypeError(msg.format(type(ext)))

        deps = ext.get_dependencies()
        if deps.difference(features_from_sorted):
            if cnt + 1 > retry:
                msg = "Maximun retry ({}) to sort achieved from extractor {}."
                raise RuntimeError(msg.format(retry, type(ext)))
            pending.append((ext, cnt + 1))
        else:
            sorted_ext.append(ext)
            features_from_sorted.update(ext.get_features())
    return tuple(sorted_ext)
