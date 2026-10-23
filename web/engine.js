/* Gofer in the browser: the Go engine (rules + search, compiled to WebAssembly)
   evaluating positions with the real network through onnxruntime-web.
   Everything is fetched on first use: loadGofer(onStatus) -> Promise<gofer>.

   gofer.newGame(komi)          -> state
   gofer.play(point)            -> {ok, error?, state}     point = y*9+x, -1 = pass
   gofer.undo(n)                -> state
   gofer.genmove(visits, onProgress?) -> Promise<{move, value, visits, top, pv, state}>
   gofer.state()                -> {cells, toPlay, last, legal, gameOver, margin, capturesB, capturesW, ...}
*/
(function () {
  'use strict';
  var ORT_VERSION = '1.26.0';
  var ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@' + ORT_VERSION + '/dist/';
  var loading = null;

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = function () { reject(new Error('could not load ' + src)); };
      document.head.appendChild(s);
    });
  }

  async function load(status) {
    status('Loading the neural-network runtime…');
    if (!window.ort) await loadScript(ORT_BASE + 'ort.wasm.min.js');
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

    status('Loading the network (1.9 MB)…');
    var session = await ort.InferenceSession.create('engine/gofer-9x9.onnx', { executionProviders: ['wasm'] });

    status('Starting the engine…');
    if (!window.Go) await loadScript('engine/wasm_exec.js');
    var go = new Go();
    var ready = new Promise(function (r) { window.goferReady = r; });
    var bytes = await fetch('engine/gofer.wasm').then(function (r) {
      if (!r.ok) throw new Error('engine/gofer.wasm: HTTP ' + r.status);
      return r.arrayBuffer();
    });
    var inst = await WebAssembly.instantiate(bytes, go.importObject);
    go.run(inst.instance);
    await ready;

    window.gofer.setEvaluator(async function (spatial, global, n) {
      var out = await session.run({
        spatial: new ort.Tensor('float32', spatial, [n, 13, 9, 9]),
        global: new ort.Tensor('float32', global, [n, 4]),
      });
      return { policy: out.policy.data, value: out.value.data };
    });
    status('');
    return window.gofer;
  }

  window.loadGofer = function (onStatus) {
    if (!loading) {
      loading = load(onStatus || function () {});
      loading.catch(function () { loading = null; });
    }
    return loading;
  };
})();
