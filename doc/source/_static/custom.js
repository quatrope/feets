function setUpContent(){
    $("div#extractors .warning").addClass("alert alert-warning");
    $("div#extractors .warning").prepend("<h5>Warning<h5><hr>");

    var $liFeatures = $("ul.localtoc li").has("a[href='#The-Features']");
    $liFeatures.append("<ul id='features-list-links'></ul>");
    var $ulFeaturesList = $liFeatures.find("ul#features-list-links");

    var $allPanels = $("div.panel-group#extractors div.panel, div.panel-collapse");

    $("div.panel-group#extractors div.panel").each(function(idx, div){
        var $div = $(div);
        var $a = $div.find("a.extractor-doc");
        var $panel = $div.find("div.panel-collapse ");
        var title = $div.find("span.extractor-name").text().trim();
        var id = $div.find("div.panel-heading").attr("id");

        $ulFeaturesList.append("<li><a href='#" + id + "'>" + title + "</a></li>");
        $ulFeaturesList.find("a[href='#" + id + "']").click(function(){
            $allPanels.not($panel).collapse("hide");
            $panel.collapse('show');
            window.location.hash = "#" + id;
        });
    });
}

$(document).ready(function(){


});
