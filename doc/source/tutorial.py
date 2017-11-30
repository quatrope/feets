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

# introduction
def ts_anim():
    # create a simple animation
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 100), ylim=(-1, 1))
    Color = [ 1 ,0.498039, 0.313725];
    line, = ax.plot([], [], '*',color = Color)
    plt.xlabel("Time")
    plt.ylabel("Measurement")

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = np.linspace(0, i+1, i+1)
        ts = 5*np.cos(x * 0.02 * np.pi) * np.sin(np.cos(x)  * 0.02 * np.pi)
        line.set_data(x, ts)
        return line,

    return animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=100, interval=200, blit=True)


def macho_video():
    return YouTubeVideo('qMx4ozpSRuE',  width=750, height=360, align='right')


def macho_example11():
    picture = Image(filename='_static/curvas_ejemplos11.jpg')
    picture.size = (100, 100)
    return picture

# the library

magnitude_ex = np.random.rand(30)
time_ex = np.arange(0, 30)

# features table

FEATURES_TABLE_TEMPLATE = jinja2.Template("""
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
""")

def features_table():

    rows = []
    for feature, ext in feets.extractors.registered_extractors().items():
        if "Signature" in feature or "_harmonics_" in feature:
            continue
        row = (
            feature,
            ext.get_features().difference([feature]),
            ext.get_dependencies(),
            ext.get_data())
        rows.append(row)

    FourierComponents = feets.extractor_of("Freq2_harmonics_rel_phase_0")
    rows.append((
        "Freq{i}_harmonics_amplitude_{j}",
        ["Freq{i}_harmonics_amplitude_{j} and Freq{i}_harmonics_rel_phase_{j}"],
        FourierComponents.get_dependencies(), FourierComponents.get_data()
    ))

    return HTML(FEATURES_TABLE_TEMPLATE.render(rows=sorted(rows)))

RESULT_TABLE_TEMPLATE = jinja2.Template("""
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
""")

def as_table(features, values):
    rows = zip(features, values)
    return HTML(RESULT_TABLE_TEMPLATE.render(rows=rows))


DOC_TEMPLATE = jinja2.Template("""
<div class="panel-group" id="extractors" role="tablist" aria-multiselectable="true">
  {% for name, doc, ext, fresume, datas in rows %}
  <div class="panel panel-default">
    <div class="panel-heading" role="tab" id="heading-feature-{{ name }}">
      <h4 class="panel-title">
        <a role="button" data-toggle="collapse" data-parent="#extractors" href="#collapse-feature-{{ name }}" aria-expanded="true" aria-controls="collapse-feature-{{ name }}">
          Extractor <span class="text-info">{{ name }}</span>
        
         <span class="pull-right">
             {% for feature in fresume %}
             <span class="label lb-lg feature-resume label-default">{{ feature }}</span>
             {% endfor %}
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

""")


def features_doc():
    import feets
    
    import rst2html5_
    
    from docutils.core import publish_parts
    
    rows = []
    extractors = sorted({
        e for e in feets.registered_extractors().values()})
    for idx, ext in enumerate(extractors):
        name = ext.__name__

        doc = publish_parts(
            ext.__doc__ or name,
            writer_name='html5', 
            writer=rst2html5_.HTML5Writer())["body"]
        
        features = ext.get_features()
        data = sorted(ext.get_data(), key=feets.extractors.DATAS.index)
        
        if len(features) > 4:
            features = list(features)[:4] + ["..."]
        rows.append((name, doc, ext, features, data))
    rows.sort()
    return HTML(DOC_TEMPLATE.render(rows=rows))
        