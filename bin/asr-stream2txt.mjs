#!/usr/bin/env node

import { main } from "../src/asr-stream2txt.mjs";

await main(process.argv.slice(2), {
  cwd: process.cwd(),
  env: process.env,
  stdin: process.stdin,
  stdout: process.stdout,
  stderr: process.stderr,
  exit: (code) => {
    process.exitCode = code;
  },
  on: process.on.bind(process),
  off: process.off.bind(process),
});
