# coding=utf-8

import json
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import os
import sys
import time
from urllib.parse import urlparse, parse_qs

from zhilight import QuantConfig, QuantType, CPMBee, LLaMA
from zhilight.loader import LLaMALoader, ModelLoader

from zhilight.dynamic_batch import DynamicBatchConfig, GeneratorArg, DynamicBatchGenerator
from zhilight.dynamic_batch import StreamHandler, StreamResultType
import random
import sys

is_bee = False
HTML = """
<html>
<head>
<meta charset="UTF-8">
<title>Stream Generator Demo</title>
<script src="https://code.jquery.com/jquery-3.2.1.min.js"></script>
<style>
body {
    margin: auto;
    padding: 10px;
}
textarea {
  // resize: none;
  width : auto;
  height: auto;
  color:blue;
}
h1 { display: none; }
p { padding: 0; margin: 0px; }
div  {
  width: auto;
  max-width: 500px;
  min-height: 100px;
  border-style: solid;
  border-color: #888888;
}
</style>
</head>
<body>
<h1>Stream Generator Demo</h1>
<p><label for="genInput">Input:</label></p>
<textarea id="genInput" rows='4' cols="50"></textarea>
<p><button id='genButton' type="button">Generate</button>&nbsp;&nbsp;<span id='info'/></p>
<textarea disabled id="genResult" rows='12' cols="50"></textarea>
</body>
<script>
var task_id = '';
var gen_state = '';
function on_result(res) {
    res_text = res["output"];
    $('#genResult').text(res["type"] == 1 ? $('#genResult').text() + res_text : res_text);
    $("#genResult").css("background-color", res["type"] == 3 ? "#82f986" : "white");
    if (res["type"] == 3) {
        summary = $('#info').text();
        summary += "; OutTNum= " + res["out_token_num"];
        summary += "; TPS= " + res["tps"];
        $('#info').html(summary);
        $('#genButton').html('Generate')
        gen_state = '';
    }
}
function req_next(task_id) {
// alert(task_id);
  $.ajax({
  	url: "/next",
    method: 'POST',
    type: 'json',
    data: task_id,
    error: function(a, b, c) {
        alert( b );
    }
  }).done(function(res) {
    on_result(res);
    if (res["type"] != 3 && gen_state != 'cancel') {
        req_next(task_id);
    }
  });
}
function req_cancel(task_id) {
  console.log('Cancel task ' + task_id);
  $.ajax({
  	url: "/cancel",
    method: 'POST',
    type: 'json',
    data: task_id,
    error: function(a, b, c) {
        alert( b );
    }
  }).done(function(res) {
      console.log('Done cancel task ' + task_id);
  });
}

$('#genButton').click(function() {
  if (gen_state == 'going') {
    console.log('cancel click');
    $('#genButton').html('Generate');
    gen_state = 'cancel';
    if (task_id != '') {
      req_cancel(task_id);
    }
    return;
  }
  task_id = '';
  gen_state = 'going';
  $('#genButton').html('Cancel')
  txt = $('#genInput').val();
  if (txt == "") {
    return;
  }
  // alert(txt);
  var ts0 = Date.now();
  var myURL = new URL(window.location.toLocaleString());
  url = "/gen" + (myURL.search || "?");
  if ($('#prompt').length > 0)
    url += "&prompt=" + encodeURIComponent($('#prompt')[0].value);
  var request = $.ajax({
  	url: url,
    method: 'POST',
    type: 'json',
    data: txt,
    error: function(a, b, c) {
        alert( b );
    }
  });
  request.done(function(res) {
    var millis = Date.now() - ts0;
    summary = 'InputTNum=' + res["input_token_num"] + '; FirstTDelay=' + millis + "/" + res["firstTokenDelay"];
    $('#info').html(summary);
    on_result(res);
    task_id = res["task_id"];
    if (gen_state == 'cancel') {
        req_cancel(task_id);
    } else if (res["type"] != 3) {
        req_next(task_id);
    }
  });
});
$( document ).ready(function() {
    var myURL = new URL(window.location.toLocaleString());
    var h = myURL.searchParams.get("resHeight");
    if (h) {
        $('#genResult').height(h);
    }
    var prompt = myURL.searchParams.get("prompt");
    if (prompt && $('#prompt').length > 0) {
        $('#prompt').val(prompt);
    }
});
</script>
</html>
"""


def get_page():
    if is_bee:
        prompt_input = '<label >Bee prompt:&nbsp;</label><input id="prompt" type="text" value=""></input>'
        return HTML.replace('<label for="genInput">Input:</label>', prompt_input)
    return HTML


class DemoPageHandler(BaseHTTPRequestHandler):
    def write_page(self, content, c_type, code=HTTPStatus.OK):
        content = content.encode("utf8")
        self.send_response(code)
        self.send_header("Content-Type", c_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        # self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content)

    @staticmethod
    def build_multi_frame_page(row, col, query):
        page = "<html><title>Stream Generator Demo</title><body>" \
               "<table style='width:100%; height: 100%'>"
        for r in range(row):
            page += "<tr>"
            for c in range(col):
                page += f"<td><iframe src='/?{query}' style='width:100%; height: 100%'></iframe></td>"
            page += "</tr>\n"
        return page + "</table></body></html>"

    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            self.write_page(get_page(), "text/html")
        elif self.path.startswith("/many"):
            query = urlparse(self.path).query
            qs = parse_qs(query)
            row = int(qs.get("row", [2])[0])
            col = int(qs.get("col", [2])[0])
            self.write_page(self.build_multi_frame_page(row, col, query), "text/html")
        else:
            self.send_not_found()

    def send_not_found(self):
        self.write_page("not found", "text/plain", HTTPStatus.NOT_FOUND)


class DemoStreamHttpServer:
    def __init__(self, generator: DynamicBatchGenerator, arg: GeneratorArg = GeneratorArg()):
        generator.start()
        self.generator = generator
        self.arg = arg
        self.handler_dict = {}
        self.ts1_dict = {}

    def create_handler(self, *args, **kwargs):
        generator = self.generator
        arg = self.arg
        handler_dict = self.handler_dict
        ts1_dict = self.ts1_dict

        class DemoHttpHandler(DemoPageHandler):
            def do_POST(self):
                content_len = int(self.headers.get('Content-Length', 0))
                data = self.rfile.read(content_len).decode()
                ts0 = time.time()
                first = self.path.startswith('/gen')
                if first:
                    qs = parse_qs(urlparse(self.path).query)
                    prompt = qs.get("prompt", [""])[0]
                    arg.temperature = float(qs.get("temperature", [1.0])[0])
                    arg.top_p = float(qs.get("top_p", [1.0])[0])
                    arg.top_k = int(qs.get("top_k", [0])[0])
                    arg.seed = int(qs.get("seed", [0])[0])
                    arg.beam_size = int(qs.get("beam_size", [1])[0])
                    arg.max_length = int(qs.get("max_length", [100])[0])
                    arg.repetition_penalty = float(qs.get("repetition_penalty", [1.])[0])
                    arg.ngram_penalty = float(qs.get("ngram_penalty", [1.])[0])
                    diverse = int(qs.get("diverse", [0])[0])
                    if diverse and arg.seed == 0:
                        arg.seed = random.randint(1, 300000)
                    print(f"prompt: {prompt}, temperature={arg.temperature}, top_p={arg.top_p}, top_k={arg.top_k}, "
                          f"repetition_penalty={arg.repetition_penalty}, seed={arg.seed}, beam_size={arg.beam_size}, max_length={arg.max_length}")
                    if is_bee:
                        data = {"input": data.strip().replace('<', '<<'), "prompt": prompt, "<ans>": ""}
                        print(json.dumps(data, ensure_ascii=False))

                    stream_handler = generator.stream_generate(data, arg=arg)
                    stream_handler.ts0 = ts0
                    task_id = str(stream_handler.task_id)
                    print("task_id", task_id)
                    handler_dict[task_id] = stream_handler
                elif self.path.startswith("/cancel"):
                    task_id = data
                    stream_handler = handler_dict[task_id]
                    print(f"Cancel task {task_id}", flush=True)
                    stream_handler.cancel()
                    self.write_page("{}", "text/json")
                    return
                elif self.path.startswith("/next"):
                    task_id = data
                    stream_handler = handler_dict[task_id]
                else:
                    print("Path not found:", self.path)
                    self.send_not_found()
                    return

                flag, res_id, output, score, time_elapsed = \
                    stream_handler.decode_stream_res(stream_handler.get_result())
                while not first and stream_handler.has_result():
                    flag1, _, output1, _, _ = \
                        stream_handler.decode_stream_res(stream_handler.get_result())
                    output = (output + output1) if flag1 == StreamResultType.Incremental else output1
                    flag = max(flag1, flag)

                res = {
                    "type": flag,
                    "output": output,
                    "task_id": task_id,
                    "input_token_num": stream_handler.input_tokens_num,
                }
                if flag == StreamResultType.Final:
                    res["out_token_num"] = stream_handler.output_tokens_nums[0]
                    elapsed = time.time() - ts1_dict[task_id]
                    res["tps"] = f"{(stream_handler.output_tokens_nums[0]) / elapsed :.1f}"
                    print(f"elapsed: {elapsed} out_token_num: {res['out_token_num']}, tps: {res['tps']}")
                    del handler_dict[task_id]
                    if "<s>" in output:
                        print(stream_handler.token_cache)

                if first:
                    print(f"firstTokenDelay:{time.time() - ts0:.3f}")
                    res["firstTokenDelay"] = int(1000 * (time.time() - ts0))
                    ts1_dict[task_id] = time.time()
                content = json.dumps(res, ensure_ascii=False)
                self.write_page(content, "text/json")

            def log_message(self, format, *args):
                if False:
                    sys.stderr.write("%s - - [%s] %s\n" %
                                     (self.address_string(),
                                      self.log_date_time_string(),
                                      format%args))

        return DemoHttpHandler(*args, **kwargs)

    def run(self, port=8008):
        server_address = ('', port)
        httpd = ThreadingHTTPServer(server_address, self.create_handler)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def load_model(model_path):
    nbytes = sum(d.stat().st_size for d in os.scandir(model_path) if d.is_file())
    if nbytes < (20 << 30):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not "awq" in model_path.lower():
        os.environ["AWQ_USE_EXLLAMA"] = "0"
    model_config = LLaMALoader.load_llama_config(model_path)
    if "minicpm" in model_path:
        model_config["model_type"] = "cpm_dragonfly"
    parallel = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")) > 1
    print("parallel:", parallel)

    t0 = time.time()
    model = LLaMA(
        f"{model_path}",
        model_config=model_config,
        quant_config=None,
        parallel=parallel,
    )
    model.load_model_pt(model_path)
    print(f">>>Load model '{model_path}' finished in {time.time() - t0:.2f} seconds<<<")
    return model


def main(model_path):
    model = load_model(model_path)

    dyn_config = DynamicBatchConfig(max_batch=20, keep_eos=True,)
    with DynamicBatchGenerator(dyn_config, model) as generator:
        DemoStreamHttpServer(generator, GeneratorArg(ngram_penalty=1., max_length=1000)).run()


if __name__ == "__main__":
    main(sys.argv[1])
