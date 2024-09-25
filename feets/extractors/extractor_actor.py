from .extractor import ExtractorContractError


class ExtractorActor:
    def __init__(self, extractor, **extractor_params):
        self.extractor = extractor(**extractor_params)

    def preprocess_arguments(self, data, dependencies):
        kwargs = {}

        # add the required features
        for d in self.extractor.get_dependencies():
            kwargs[d] = dependencies[d]

        # add the required data
        for d in self.extractor.get_data():
            kwargs[d] = data[d]

        return kwargs

    def validate_result(self, result):
        if result is None:
            result = dict()

        # validate if the extractor generates the expected features
        expected_features = self.extractor.get_features()
        if expected_features is None:
            expected_features = frozenset()

        diff = set(result).symmetric_difference(expected_features)
        if diff:
            cls_name = type(self).__qualname__
            estr, fstr = ", ".join(expected_features), ", ".join(result.keys())
            raise ExtractorContractError(
                f"The extractor '{cls_name}' expected the features {estr}. "
                f"Found: {fstr!r}"
            )

        return result

    def select_extract_and_validate(self, data, dependencies):
        extract_kwargs = self.preprocess_arguments(data, dependencies)

        results = self.extractor.extract(**extract_kwargs)

        extracted_features = self.validate_result(results)

        return extracted_features

    def get_feature(self, extracted_features, feature):
        return extracted_features[feature]
