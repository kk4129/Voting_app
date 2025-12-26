require("@nomicfoundation/hardhat-toolbox");

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: "0.8.9",
  networks: {
    hardhat: {
      chainId: 1337
    },
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 1337
    }
  },
  paths: {
    artifacts: './frontend/src/artifacts', // Direct artifacts to frontend
    cache: './cache',
    sources: './contracts',
    tests: './test',
  },
  mocha: {
    timeout: 20000 // Set timeout for tests
  },
  // Add any plugins here if needed
};