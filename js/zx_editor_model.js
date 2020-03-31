// When the next line is uncommented, the module is reloaded every time the javascript is imported
// This is useful for development.
require.undef('zx_editor');

define('zx_editor', ["@jupyter-widgets/base", "make_editor"], function(widgets,make_editor) {
    console.log("Loading model script");
    var ZXEditorModel = widgets.DOMWidgetModel.extend({
        defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
            _model_name: 'ZXEditorModel',
            _view_name: 'ZXEditorView',
            _model_module: 'zx_editor',
            _view_module: 'zx_editor',
            _model_module_version: '0.1.0',
            _view_module_version: '0.1.0',
            graph_json: '{"nodes": [], "links": []}',
            graph_id: '0',
            graph_width: 600.0,
            graph_height: 400.0,
            graph_node_size: 5.0
        })
    });
    
    var ZXEditorView = widgets.DOMWidgetView.extend({
        
        render: function() {
            var btn = document.createElement('button')
            btn.textContent = 'Push changes'
            btn.onclick = this.value_changed.bind(this);
            this.el.appendChild(btn)
            var mydiv = document.createElement('div');
            mydiv.setAttribute('style', 'overflow:auto');
            var div_id = 'graph-interactive-' + this.model.get('graph_id');
            mydiv.setAttribute('id', div_id);
            this.el.appendChild(mydiv);
            this.graphData = {};
            this.graphData.graph = JSON.parse(this.model.get('graph_json'));
            this.graphData.max_name = make_editor.prepareGraph(this.graphData.graph);
            this.graphData.width = this.model.get("graph_width");
            this.graphData.height = this.model.get("graph_height");
            this.graphData.node_size = this.model.get("graph_node_size");
            this.graphData.update_graph = make_editor.showGraph(mydiv, this.graphData, false, false);
            //this.model.on('change:value', this.value_changed, this);
            this.listenTo(this.model, 'change:graph_json', this.graph_changed, this);
        },
        
        value_changed: function() {
            console.log("Pushing changes to kernel")
            this.model.set('graph_json', JSON.stringify(this.strip_graph()));
            this.model.save_changes();
            //this.model.touch();
        },

        strip_graph: function() {
            var g = {links: [], nodes: []}
            this.graphData.graph.nodes.forEach(function(d) {
                g.nodes.push({"name": d.name, "x":d.x, "y": d.y, "t": d.t, "phase": d.phase})
            });
            this.graphData.graph.links.forEach(function(d) {
               g.links.push({"source": d.source.name, "target": d.target.name, "t":d.t}) 
            });
            //console.log(g);
            return g
        },

        graph_changed: function() {
            console.log("Updating graph");
            var new_graph = JSON.parse(this.model.get('graph_json'));
            this.graphData.max_name = make_editor.prepareGraph(new_graph);
            this.graphData.graph = new_graph;
            console.log(this.graphData.graph);
            this.graphData.update_graph();
        }
    });
    
    return {
        ZXEditorModel: ZXEditorModel,
        ZXEditorView: ZXEditorView
    };
});