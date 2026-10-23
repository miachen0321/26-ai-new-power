import sonarjs from "eslint-plugin-sonarjs";

const shared = {
  name: "quality-thresholds",
  plugins: { sonarjs },
  rules: {
    "complexity": ["error", { max: 10 }],
    "max-depth": ["error", { max: 3 }],
    "max-lines-per-function": ["error", { max: 50, skipBlankLines: false, skipComments: true }],
    "sonarjs/cognitive-complexity": ["error", 15],
  },
};

const browserFiles = {
  name: "browser-docs",
  files: ["docs/**/*.js"],
  languageOptions: {
    globals: {
      window: "readonly",
      document: "readonly",
      fetch: "readonly",
      location: "readonly",
      setTimeout: "readonly",
      clearTimeout: "readonly",
    },
  },
  rules: shared.rules,
  plugins: shared.plugins,
};

const nodeFiles = {
  name: "node-scripts-tests",
  files: ["scripts/**/*.js", "tests/**/*.js"],
  languageOptions: {
    globals: {
      require: "readonly",
      module: "writable",
      exports: "writable",
      global: "readonly",
      process: "readonly",
      __dirname: "readonly",
      __filename: "readonly",
    },
  },
  rules: shared.rules,
  plugins: shared.plugins,
};

const config = [shared, browserFiles, nodeFiles];
export default config;
