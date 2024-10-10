import feets

fs = feets.FeatureSpace(only=["Mean", "Std", ])

lc = feets.datasets.load_MACHO_example()

feats = fs.extract(**dict(lc.data.B))
feats = feets.core.Features(features=[feats.features[0]] * 500, extractors=feats.extractors)


extractors = []

with joblib.Parallel() as p:
    delayeds_exts = list(map(joblib.delayed, extractors))
    for lc in lcs:
        results = p(extract(lc) for extract in delayeds_exts)





