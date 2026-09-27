'use strict';

/**
 * React Native AFM-D engine binding.
 * Native module should link libaria-engine_ffi and expose systemone(JSON).
 * This JS layer mirrors the TypeScript API shape for host tests.
 */

class AriaEngine {
  constructor(checkpoint, track = 'encoder') {
    this.checkpoint = checkpoint;
    this.track = track;
  }

  async systemone(request) {
    throw new Error(
      'React Native native module not linked; embed libaria-engine_ffi and bridge aria_systemone'
    );
  }

  destroy() {}
}

module.exports = { AriaEngine };
