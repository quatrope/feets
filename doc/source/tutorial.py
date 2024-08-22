# Based on: http://isadoranun.github.io/tsfeat/FeaturesDocumentation.html

# JSAnimation import available at https://github.com/jakevdp/JSAnimation

import warnings

import numpy as np

import jinja2

from matplotlib import pyplot as plt
from matplotlib import animation

from JSAnimation import IPython_display

from IPython.display import Image, HTML, YouTubeVideo

warnings.simplefilter("ignore", FutureWarning)

import feets

warnings.simplefilter("ignore", feets.ExtractorWarning)


# introduction
def ts_anim():
    # create a simple animation
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 100), ylim=(-1, 1))
    Color = [1, 0.498039, 0.313725]
    (line,) = ax.plot([], [], "*", color=Color)
    plt.xlabel("Time")
    plt.ylabel("Measurement")

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        x = np.linspace(0, i + 1, i + 1)
        ts = 5 * np.cos(x * 0.02 * np.pi) * np.sin(np.cos(x) * 0.02 * np.pi)
        line.set_data(x, ts)
        return (line,)

    return animation.FuncAnimation(
        fig, animate, init_func=init, frames=100, interval=200, blit=True
    )


def macho_video():
    return YouTubeVideo("qMx4ozpSRuE", width=750, height=360, align="right")


def macho_example11():
    picture = Image(filename="_static/curvas_ejemplos11.jpg")
    picture.size = (100, 100)
    return picture


# the library

magnitude_ex = np.random.rand(30)
time_ex = np.arange(0, 30)

# features table

FEATURES_TABLE_TEMPLATE = jinja2.Template(
    """
<table class="table-condensed">
    <thead>
        <th>Feature</th>
        <th>Computed with</th>
        <th>Dependencies</th>
        <th>Input Data</th>
    </thead>
    <tbody>
        {% for feat, cwith, deps, input in rows %}
        <tr>
            <td>{{ feat }}</td>
            <td>{{ ", ".join(cwith or []) }}</td>
            <td>{{ ", ".join(deps or []) }}</td>
            <td>{{ ", ".join(input or []) }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
"""
)


def features_table():

    rows = []
    for feature, ext in feets.extractors.registered_extractors().items():
        if "Signature" in feature or "_harmonics_" in feature:
            continue
        row = (
            feature,
            ext.get_features().difference([feature]),
            ext.get_dependencies(),
            ext.get_data(),
        )
        rows.append(row)

    FourierComponents = feets.extractor_of("Freq2_harmonics_rel_phase_0")
    rows.append(
        (
            "Freq{i}_harmonics_amplitude_{j}",
            [
                "Freq{i}_harmonics_amplitude_{j} and Freq{i}_harmonics_rel_phase_{j}"
            ],
            FourierComponents.get_dependencies(),
            FourierComponents.get_data(),
        )
    )

    return HTML(FEATURES_TABLE_TEMPLATE.render(rows=sorted(rows)))


RESULT_TABLE_TEMPLATE = jinja2.Template(
    """
<table class="table-condensed">
    <thead>
        <th>Feature</th>
        <th>Value</th>
    </thead>
    <tbody>
        {% for k, v in rows %}
        <tr>
            <td>{{ k }}</td>
            <td>{{ v }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
"""
)


def as_table(features, values):
    rows = zip(features, values)
    return HTML(RESULT_TABLE_TEMPLATE.render(rows=rows))


DOC_TEMPLATE = jinja2.Template(
    """
<div class="section" id="The-Features">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>
<div class="panel-group" id="extractors" role="tablist" aria-multiselectable="true">
  {% for name, doc, ext, fresume, datas in rows %}
  <div class="panel panel-default">
    <div class="panel-heading" role="tab" id="heading-feature-{{ name }}">
      <h4 class="panel-title">
        <a class="extractor-doc" role="button" data-toggle="collapse" data-parent="#extractors" href="#collapse-feature-{{ name }}" aria-expanded="true" aria-controls="collapse-feature-{{ name }}">
          <span class="extractor-name">Extractor <span class="text-info">{{ name }}</span></span>

          <span class="pull-right">

              {% if not ext.has_warnings() %}
              <span class="label lb-lg feature-resume label-warning">Warning</span>
              {% endif %}

             {% for feature in fresume %}
             <span class="label lb-lg feature-resume label-default">{{ feature }}</span>
             {% endfor %}

             {% if not doc %}
             <span class="label lb-lg feature-resume label-danger">No Doc</span>
             {% endif %}
         </span>
         </a>

      </h4>
    </div>
    <div id="collapse-feature-{{ name }}" class="panel-collapse collapse" role="tabpanel" aria-labelledby="heading-feature-{{ name }}">
      <div class="panel-body">
         <p>{{ doc }}</p>
        <h5>Required Data</h5>
         <div>
             {% for data in datas %}
                 <span class="label label-info">{{ data }}</span>
             {% endfor %}
         </div>

         <h5>Full list of features</h5>
         <div>
             {% for feature in ext.get_features() %}
             <span class="label label-default">{{ feature }}</span>
             {% endfor %}
         </div>

        <h5>Parameters</h5>
        <div class="row">
        <div class="col-md-10">
            {% for k, v in ext.get_default_params().items() %}
            <div class="input-group">
                <span class="input-group-addon" id="basic-addon1">{{k}}</span>
                <span type="text" class="form-control"
                    aria-label="{{k}}" aria-describedby="basic-addon1">
                    <code>{{v}}</code></span>
            </div>
            {% else %}
            -
            {% endfor %}
        </div>
        </div>

         <h5>Dependencies</h5>
         <div>
         {% for dep in ext.get_dependencies() %}
            <span class="label label-warning">{{ dep }}</span>
         {% else %}
             -
         {% endfor %}
         </div>
      </div>
    </div>
  </div>
  {% endfor %}
</div>
<script>
$("div#extractors .warning").addClass("alert alert-warning");
$("div#extractors .warning").prepend("<h5 class='text-warning'>Warning<h5><hr>");
</script>
</div>
"""
)


def deindent_reference(string):
    lines = string.splitlines()
    to_remove = []
    to_bold = []
    for idx, l in enumerate(lines):
        if l.strip() and (
            not l.replace("-", "").strip() or not l.replace("=", "").strip()
        ):
            to_remove.append(idx)
            to_bold.append(idx - 1)

    deindented = []
    for idx, l in enumerate(lines):
        if idx in to_remove:
            continue
        elif idx in to_bold:
            l = "**{}**".format(l.strip())
        deindented.append(l)

    return "\n".join(deindented)


def make_title(name):
    name = "{} Extactor".format(name)
    dec = "=" * len(name)
    return "{}\n{}\n{}\n".format(dec, name, dec)


def features_doc():
    import feets

    import rst2html5_

    from docutils.core import publish_parts

    rows = []
    extractors = sorted({e for e in feets.registered_extractors().values()})
    for idx, ext in enumerate(extractors):
        name = ext.__name__

        doc = publish_parts(
            make_title(name) + deindent_reference(ext.__doc__ or ""),
            writer_name="html5",
            writer=rst2html5_.HTML5Writer(),
        )["body"]

        features = ext.get_features()
        data = sorted(ext.get_data(), key=feets.extractors.DATAS.index)

        if len(features) > 4:
            features = list(features)[:4] + ["..."]
        rows.append((name, doc, ext, features, data))
    rows.sort()
    return HTML(DOC_TEMPLATE.render(rows=rows))
