import ray


class ExtractorActor:
    def __init__(self, extractor, result_refs_by_dependency):
        self._extractor = extractor
        self._refs = result_refs_by_dependency

    def get_dependencies_from_refs(self):
        dependencies = {}

        for dependency in self._extractor.get_dependencies():
            result_ref = self._refs[dependency]
            result = ray.get(result_ref)
            dependencies[dependency] = result[dependency]

        return dependencies

    def extract(self, data):
        dependencies = self.get_dependencies_from_refs()

        features = self._extractor.select_extract_and_validate(
            data, dependencies
        )

        return features
