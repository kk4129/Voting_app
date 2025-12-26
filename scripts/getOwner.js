const hre = require("hardhat");
async function main() {
  const addr = require("../frontend/src/artifacts/contracts/contract-address.json").Voting;
  const ctr = await hre.ethers.getContractAt("Voting", addr);
  const owner = await ctr.owner();
  console.log("Contract owner:", owner);
}
main().catch(e => { console.error(e); process.exit(1); });