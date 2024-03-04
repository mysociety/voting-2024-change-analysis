from __future__ import annotations

from typing import NamedTuple, Protocol


class ScoreFloatPair(NamedTuple):
    """
    Group the counts of strong and weak votes to have less variables to pass around.
    """

    weak: float
    strong: float

    def add(self, other: ScoreFloatPair) -> ScoreFloatPair:
        return ScoreFloatPair(
            weak=self.weak + other.weak,
            strong=self.strong + other.strong,
        )

    def divide(self, other: float) -> ScoreFloatPair:
        return ScoreFloatPair(
            weak=self.weak / other,
            strong=self.strong / other,
        )


class ScoringFuncProtocol(Protocol):
    """
    Set a protocol for scoring functions to validate
    correct order of arguments and return type.
    """

    @staticmethod
    def score(
        *,
        votes_same: ScoreFloatPair,
        votes_different: ScoreFloatPair,
        votes_absent: ScoreFloatPair,
        votes_abstain: ScoreFloatPair,
        agreements_same: ScoreFloatPair,
        agreements_different: ScoreFloatPair,
    ) -> float:
        ...


class PublicWhipScore(ScoringFuncProtocol):
    @staticmethod
    def score(
        *,
        votes_same: ScoreFloatPair,
        votes_different: ScoreFloatPair,
        votes_absent: ScoreFloatPair,
        votes_abstain: ScoreFloatPair,
        agreements_same: ScoreFloatPair,
        agreements_different: ScoreFloatPair,
    ) -> float:
        """
        Calculate the classic Public Whip score for a difference between two MPs.

        The score is a number between 0 and 1, where 0 is a perfect match and 1 is a perfect
        mismatch. Returns -1 if there are no votes to compare.

        This assumes two kinds of votes: weak and strong.

        weak votes are worth a base score of 10/10 points if aligned, 0/10 points if not aligned, and 1/2 points if absent.
        Strong votes are worth a base score of 50/50 points if aligned, 0/50 points if not aligned, and 25/50 points if absent.

        The weird bit of complexity here is absences on weak votes reduce the total of the comparison.
        This means that MPs are only lightly penalised for missing votes if they attended some votes, or if there are strong votes.
        If all votes are weak and absent, the score will be 0.5.

        So if there were five weak votes, two in line with the policy, and three absent - the difference would be 0.12.
        But if weak votes were treated the same way as strong votes (5/10) - the difference would be 0.3.

        So the practical result of making a policy a mix of strong and weak votes is first,
        obviously that weak votes make up a smaller part of the total score.
        But the second is that strong votes penalise absences more than weak votes.

        Strong votes were originally intended to reflect three line whips, but in practice have broadened out to mean 'more important'.

        Do nothing with agreements for the moment.

        """
        vote_weight = ScoreFloatPair(weak=10.0, strong=50.0)
        absence_total_weight = ScoreFloatPair(weak=2.0, strong=50.0)

        absence_weight = ScoreFloatPair(weak=1.0, strong=25.0)

        # treat abstentions as absences
        votes_absent_or_abstain = votes_absent.add(votes_abstain)

        points = (
            vote_weight.weak * votes_different.weak
            + vote_weight.strong * votes_different.strong
            + absence_weight.weak * votes_absent_or_abstain.weak
            + (
                (absence_weight.strong) * votes_absent_or_abstain.strong
            )  # Absences on strong votes are worth half the strong value
        )

        avaliable_points = (
            vote_weight.weak * votes_same.weak
            + vote_weight.weak * votes_different.weak
            + vote_weight.strong * votes_same.strong
            + vote_weight.strong * votes_different.strong
            + absence_total_weight.strong * votes_absent_or_abstain.strong
            + absence_total_weight.weak * votes_absent_or_abstain.weak
        )  # Absences on weak votes reduce the total of the comparison

        # are there just absences - regardless of this would score, twfy ignores it so let's do so here
        all_absences = votes_absent.weak + votes_absent.strong
        all_others = (
            votes_same.weak
            + votes_same.strong
            + votes_different.weak
            + votes_different.strong
            + votes_abstain.weak
            + votes_abstain.strong
        )

        if all_others == 0 and all_absences > 0:
            return -1

        if avaliable_points == 0:
            return -1

        return points / avaliable_points


class SimplifiedGradiatedScore(ScoringFuncProtocol):
    @staticmethod
    def score(
        *,
        votes_same: ScoreFloatPair,
        votes_different: ScoreFloatPair,
        votes_absent: ScoreFloatPair,
        votes_abstain: ScoreFloatPair,
        agreements_same: ScoreFloatPair,
        agreements_different: ScoreFloatPair,
    ) -> float:
        """
        This is a simplified version of the public whip scoring system.
        This function is used to map the differences in scores as part of analysis to the new simplified system.
        Weak weight maintain their 1/5th value

        Absences do not move the needle.
        Abstensions are recorded as present - but half the value of a weak vote.

        Strong agreements are counted the same as votes.

        """

        vote_weight = ScoreFloatPair(weak=10.0, strong=50.0)
        agreement_weight = vote_weight
        abstain_total_weight = vote_weight
        abstain_weight = vote_weight.divide(2)  # abstain is half marks
        absence_weight = ScoreFloatPair(
            weak=0.0, strong=0.0
        )  # absences are worth nothing
        absence_total_weight = ScoreFloatPair(weak=0.0, strong=0.0)  # out of nothing

        points = (
            vote_weight.weak * votes_different.weak
            + vote_weight.strong * votes_different.strong
            + absence_weight.weak * votes_absent.weak
            + absence_weight.strong * votes_absent.strong
            + abstain_weight.weak * votes_abstain.weak
            + abstain_weight.strong * votes_abstain.strong
            + agreement_weight.weak * agreements_different.weak
            + agreement_weight.strong * agreements_different.strong
        )

        avaliable_points = (
            vote_weight.weak * (votes_same.weak + votes_different.weak)
            + vote_weight.strong * (votes_same.strong + votes_different.strong)
            + agreement_weight.weak * (agreements_same.weak + agreements_different.weak)
            + agreement_weight.strong
            * (agreements_same.strong + agreements_different.strong)
            + absence_total_weight.weak * votes_absent.weak
            + absence_total_weight.strong * votes_absent.strong
            + abstain_total_weight.weak * votes_abstain.weak
            + abstain_total_weight.strong * votes_abstain.strong
        )

        if avaliable_points == 0:
            return -1

        score = points / avaliable_points

        total = (
            votes_same.add(votes_different).add(votes_absent).add(votes_abstain).strong
        )

        # if more than one absent vote cap the score to prevent a consistently
        if votes_absent.strong > 1:
            if score <= 0.05:
                score = 0.06
            elif score >= 0.95:
                score = 0.94

        # if more than one-third absent vote cap the score to prevent an 'almost always'
        if votes_absent.strong >= total / 3:
            if score <= 0.15:
                score = 0.16
            elif score >= 0.85:
                score = 0.84

        return score


class SimplifiedScore(ScoringFuncProtocol):
    @staticmethod
    def score(
        *,
        votes_same: ScoreFloatPair,
        votes_different: ScoreFloatPair,
        votes_absent: ScoreFloatPair,
        votes_abstain: ScoreFloatPair,
        agreements_same: ScoreFloatPair,
        agreements_different: ScoreFloatPair,
    ) -> float:
        """
        This is a simplified version of the public whip scoring system.
        Weak weight votes are 'informative' only, and have no score.

        Absences do not move the needle.
        Abstensions are recorded as present - but half the value of a weak vote.

        Strong agreements are counted the same as votes.

        """

        vote_weight = ScoreFloatPair(weak=0.0, strong=10.0)
        agreement_weight = vote_weight
        abstain_total_weight = vote_weight
        abstain_weight = vote_weight.divide(2)  # abstain is half marks
        absence_weight = ScoreFloatPair(
            weak=0.0, strong=0.0
        )  # absences are worth nothing
        absence_total_weight = ScoreFloatPair(weak=0.0, strong=0.0)  # out of nothing

        points = (
            vote_weight.weak * votes_different.weak
            + vote_weight.strong * votes_different.strong
            + absence_weight.weak * votes_absent.weak
            + absence_weight.strong * votes_absent.strong
            + abstain_weight.weak * votes_abstain.weak
            + abstain_weight.strong * votes_abstain.strong
            + agreement_weight.weak * agreements_different.weak
            + agreement_weight.strong * agreements_different.strong
        )

        avaliable_points = (
            vote_weight.weak * (votes_same.weak + votes_different.weak)
            + vote_weight.strong * (votes_same.strong + votes_different.strong)
            + agreement_weight.weak * (agreements_same.weak + agreements_different.weak)
            + agreement_weight.strong
            * (agreements_same.strong + agreements_different.strong)
            + absence_total_weight.weak * votes_absent.weak
            + absence_total_weight.strong * votes_absent.strong
            + abstain_total_weight.weak * votes_abstain.weak
            + abstain_total_weight.strong * votes_abstain.strong
        )

        if avaliable_points == 0:
            return -1

        score = points / avaliable_points

        total = (
            votes_same.add(votes_different).add(votes_absent).add(votes_abstain).strong
        )

        # if more than one absent vote cap the score to prevent a consistently
        if votes_absent.strong > 1:
            if score <= 0.05:
                score = 0.06
            elif score >= 0.95:
                score = 0.94

        # if more than one-third absent vote cap the score to prevent an 'almost always'
        if votes_absent.strong >= total / 3:
            if score <= 0.15:
                score = 0.16
            elif score >= 0.85:
                score = 0.84

        return score
