import { expect } from "chai";
import { ethers } from "hardhat";

describe("Voting Contract", function () {
    let Voting;
    let voting;
    let owner;
    let voter1;
    let voter2;

    beforeEach(async function () {
        Voting = await ethers.getContractFactory("Voting");
        [owner, voter1, voter2] = await ethers.getSigners();
        voting = await Voting.deploy();
        await voting.deployed();
    });

    it("Should register a voter", async function () {
        await voting.registerVoter(voter1.address);
        expect(await voting.isRegistered(voter1.address)).to.be.true;
    });

    it("Should not allow double registration", async function () {
        await voting.registerVoter(voter1.address);
        await expect(voting.registerVoter(voter1.address)).to.be.revertedWith("Voter is already registered");
    });

    it("Should allow a registered voter to vote", async function () {
        await voting.registerVoter(voter1.address);
        await voting.connect(voter1).vote(1); // Assuming 1 is a valid candidate ID
        expect(await voting.hasVoted(voter1.address)).to.be.true;
    });

    it("Should not allow a voter to vote twice", async function () {
        await voting.registerVoter(voter1.address);
        await voting.connect(voter1).vote(1);
        await expect(voting.connect(voter1).vote(1)).to.be.revertedWith("Voter has already voted");
    });

    it("Should tally votes correctly", async function () {
        await voting.registerVoter(voter1.address);
        await voting.registerVoter(voter2.address);
        await voting.connect(voter1).vote(1);
        await voting.connect(voter2).vote(1);
        const results = await voting.tallyVotes();
        expect(results[1]).to.equal(2); // Assuming candidate ID 1 received 2 votes
    });
});