const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  // pick signer by index (0 = first account, 1 = second, ...)
  const signers = await hre.ethers.getSigners();
  const deployer = signers[1]; // <-- change index to choose account

  console.log("Deploying from:", deployer.address);

  const VotingFactory = await hre.ethers.getContractFactory("Voting", deployer);
  const voting = await VotingFactory.deploy(); // pass constructor args if needed
  await voting.deployed();

  console.log("Voting deployed to:", voting.address);

  const contractDir = path.join(__dirname, "..", "frontend", "src", "artifacts", "contracts");
  if (!fs.existsSync(contractDir)) fs.mkdirSync(contractDir, { recursive: true });

  fs.writeFileSync(path.join(contractDir, "contract-address.json"), JSON.stringify({ Voting: voting.address }, null, 2));
  const artifact = await hre.artifacts.readArtifact("Voting");
  fs.writeFileSync(path.join(contractDir, "Voting.json"), JSON.stringify(artifact, null, 2));

  console.log("Frontend artifacts written to:", contractDir);
}

main().catch((e)=>{ console.error(e); process.exit(1); });