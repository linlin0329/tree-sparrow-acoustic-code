"""发布分数条件的单一十进制实现；模型分数不是校准概率。"""

from decimal import Decimal, InvalidOperation


class ScoreRule:
    def __init__(self, comparison="gt", threshold="0.5"):
        self.comparison = comparison
        self.threshold = Decimal(str(threshold))
        if (
            comparison not in ("gt", "ge")
            or not self.threshold.is_finite()
            or not Decimal(0) <= self.threshold <= Decimal(1)
        ):
            raise ValueError("分数条件须为gt/ge及[0,1]内有限十进制阈值")
        self.working_pool = comparison == "gt" and self.threshold == Decimal("0.5")
        self.selected_scope = (
            "selected_above_05"
            if self.working_pool
            else (
                "selected_at_or_above_threshold"
                if comparison == "ge"
                else "selected_above_threshold"
            )
        )
        self.excluded_scope = (
            "excluded_below_threshold" if comparison == "ge" else "excluded_at_or_below"
        )
        self.exclusion_reason = (
            "target_max_below_threshold"
            if comparison == "ge"
            else "target_max_at_or_below_threshold"
        )

    def accepts(self, score):
        try:
            value = Decimal(str(score))
        except InvalidOperation:
            return False
        if not value.is_finite() or not Decimal(0) <= value <= Decimal(1):
            return False
        return (
            value >= self.threshold
            if self.comparison == "ge"
            else value > self.threshold
        )

    def scope(self, score_status, score):
        if score_status == "valid_target_maximum":
            return self.selected_scope if self.accepts(score) else self.excluded_scope
        return (
            self.excluded_scope
            if score_status == "no_target_detection_row"
            else "unknown"
        )

    def as_dict(self):
        return dict(
            comparison=self.comparison,
            threshold=str(self.threshold),
            arithmetic="exact_decimal_no_epsilon",
        )

    @classmethod
    def from_policy(cls, policy):
        selection = policy.get("score_selection", {})
        return cls(selection.get("comparison", "gt"), selection.get("threshold", "0.5"))
