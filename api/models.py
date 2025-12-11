"""
Database models for token scam detection system.
Designed based on the provided DB schema specification.
"""

from django.db import models
from django.utils import timezone
from decimal import Decimal


class TokenInfo(models.Model):
    """
    Core model storing token metadata and pair information.
    Acts as the central reference point for all analysis data.
    Primary key: token_addr_idx (auto-generated)
    """
    # token_addr_idx is auto-generated as primary key (id field)
    token_addr = models.CharField(
        max_length=42,
        unique=True,
        db_index=True,
        help_text="Actual token contract address"
    )
    is_processing = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Flag indicating if this token is currently being analyzed"
    )
    pair_addr = models.CharField(
        max_length=42,
        help_text="Pair contract address"
    )
    pair_creator = models.CharField(
        max_length=42
    )
    token_create_ts = models.DateTimeField(
        help_text="Token creation timestamp"
    )
    lp_create_ts = models.DateTimeField(
        help_text="Liquidity pool creation timestamp"
    )
    pair_idx = models.IntegerField(
        help_text="Token index in pair (0 or 1)"
    )
    pair_type = models.CharField(
        max_length=50,
        help_text="Router type used by the pair"
    )
    token_creator_addr = models.CharField(
        max_length=42,
        blank=True,
        null=True,
        help_text="Token creator address"
    )
    symbol = models.CharField(
        max_length=20,
        blank=True,
        null=True,
        help_text="Token symbol"
    )
    name = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Token name"
    )
    holder_cnt = models.IntegerField(
        blank=True,
        null=True,
        help_text="Total number of token holders"
    )

    class Meta:
        db_table = 'token_info'
        ordering = ['-id']

    def __str__(self):
        return f"Token {self.id}: {self.token_addr}"


class AnalysisJob(models.Model):
    """
    Tracks analysis job status and progress.
    Links to TokenInfo for job-token association.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('collecting_token', 'Collecting Token Info'),
        ('collecting_pair', 'Collecting Pair Events'),
        ('collecting_holder', 'Collecting Holder Info'),
        ('preprocessing', 'Preprocessing'),
        ('analyzing_honeypot_da', 'Honeypot Dynamic Analysis'),
        ('analyzing_honeypot_ml', 'Honeypot ML Analysis'),
        ('analyzing_exit_ml', 'Exit Scam ML Analysis'),
        ('aggregating', 'Aggregating Results'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    token_addr = models.CharField(
        max_length=42,
        db_index=True,
        help_text="Token address being analyzed"
    )
    token_info = models.ForeignKey(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='jobs',
        null=True,
        blank=True,
        help_text="Link to TokenInfo after collection"
    )

    status = models.CharField(
        max_length=30,
        choices=STATUS_CHOICES,
        default='pending'
    )
    current_step = models.CharField(
        max_length=100,
        blank=True,
        help_text="Current processing step description"
    )

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(blank=True, null=True)

    error_message = models.TextField(blank=True, null=True)
    error_step = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        db_table = 'analysis_job'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['token_addr', '-created_at']),
            models.Index(fields=['status', '-created_at']),
        ]

    def __str__(self):
        return f"Job {self.id}: {self.token_addr} ({self.status})"


class Result(models.Model):
    """
    Final analysis result table.
    Stores aggregated scam detection results and risk scores.
    """
    # id field is auto-generated (separate from token_addr_idx)
    token_addr = models.CharField(
        max_length=42,
        unique=True,
        db_index=True,
        help_text="Analyzed token contract address",
    )
    created_at = models.DateTimeField(
        default=timezone.now,
        help_text="Analysis completion timestamp",
    )

    is_unformed_lp = models.IntegerField(
        default = 0
    )

    risk_score = models.JSONField(
        null=True,
        help_text="Final scam score calculated after analysis"
    )

    scam_types = models.CharField(
        max_length=100,
        help_text="Detected scam categories"
    )

    exitInsight = models.JSONField(
    )

    honeypotMlInsight = models.CharField(
        max_length=100
    )

    honeypotDaInsight = models.JSONField(
    )

        # 🔹 분석 시점 TokenInfo 스냅샷
    token_snapshot = models.JSONField(
        null=True,
        blank=True,
        help_text="Snapshot of TokenInfo at analysis time (symbol, name, pair_addr, holder_cnt ...)",
    )

    # 🔹 분석 시점 HolderInfo 요약 스냅샷
    holder_snapshot = models.JSONField(
        null=True,
        blank=True,
        help_text="Snapshot of holders at analysis time (total_holders, top holders, etc.)",
    )

    class Meta:
        db_table = 'result'
        ordering = ['-created_at']

    def __str__(self):
        return f"Result {self.id}: {self.token_addr} (score: {self.risk_score})"


class PairEvent(models.Model):
    """
    Stores raw pair event data collected from blockchain.
    Contains Mint, Burn, Sync, Swap events from the pair contract.
    """
    token_info = models.ForeignKey(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='pair_events',
        db_column='token_addr_idx'
    )

    timestamp = models.DateTimeField(
        db_index=True,
        help_text="Event timestamp (ISO format)"
    )
    block_number = models.IntegerField(
        help_text="Block number of the transaction"
    )
    tx_hash = models.CharField(
        max_length=66,
        help_text="Transaction hash"
    )
    tx_from = models.CharField(
        max_length=42,
        help_text="Transaction sender address"
    )
    tx_to = models.CharField(
        max_length=42,
        help_text="Transaction recipient address"
    )
    evt_idx = models.IntegerField(
        help_text="Event log index in transaction"
    )
    evt_type = models.CharField(
        max_length=20,
        help_text="Event type (Mint, Burn, Sync, Swap, etc.)"
    )
    evt_log = models.JSONField(
        help_text="Preprocessed event log in JSON format"
    )

    lp_total_supply = models.DecimalField(
        max_digits=78,
        decimal_places=18,
        help_text="Current LP total supply"
    )

    class Meta:
        db_table = 'pair_evt'
        ordering = ['timestamp', 'evt_idx']
        indexes = [
            models.Index(fields=['token_info', 'timestamp']),
            models.Index(fields=['block_number']),
        ]

    def __str__(self):
        return f"{self.evt_type} at {self.timestamp}"


class HolderInfo(models.Model):
    """
    Stores token holder information.
    Each row represents a single holder's balance data.
    """
    token_info = models.ForeignKey(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='holders',
        db_column='token_addr_idx'
    )

    holder_addr = models.CharField(
        max_length=42,
        help_text="Holder wallet address"
    )
    balance = models.DecimalField(
        max_digits=78,
        decimal_places=18,
        help_text="Token balance held by this address"
    )
    rel_to_total = models.CharField(
        max_length=20,
        help_text="Percentage of total supply held"
    )

    class Meta:
        db_table = 'holder_info'
        ordering = ['-balance']
        indexes = [
            models.Index(fields=['token_info', '-balance']),
        ]

    def __str__(self):
        return f"Holder {self.holder_addr}: {self.rel_to_total}"


class HoneypotProcessedData(models.Model):
    """
    Preprocessed data for honeypot detection analysis.
    One record per token after preprocessing stage.
    """
    token_info = models.OneToOneField(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='honeypot_processed',
        primary_key=True,
        db_column='token_addr_idx'
    )
    token_addr = models.CharField(
        max_length=42,
        help_text="Token contract address"
    )

    # Trade counts
    total_buy_cnt = models.IntegerField()
    total_sell_cnt = models.IntegerField()
    total_owner_sell_cnt = models.IntegerField()
    total_non_owner_sell_cnt = models.IntegerField()

    # Imbalance metrics
    imbalance_rate = models.FloatField()
    total_windows = models.IntegerField()
    windows_with_activity = models.IntegerField()

    # Event counts
    total_burn_events = models.IntegerField()
    total_mint_events = models.IntegerField()
    s_owner_count = models.IntegerField()

    # Volume metrics
    total_sell_vol = models.DecimalField(max_digits=78, decimal_places=18)
    total_buy_vol = models.DecimalField(max_digits=78, decimal_places=18)
    total_owner_sell_vol = models.DecimalField(max_digits=78, decimal_places=18)

    # Log-transformed volumes
    total_sell_vol_log = models.FloatField()
    total_buy_vol_log = models.FloatField()
    total_owner_sell_vol_log = models.FloatField()

    # Additional metrics
    liquidity_event_mask = models.IntegerField()
    max_sell_share = models.FloatField()
    unique_sellers = models.IntegerField()
    unique_buyers = models.IntegerField()
    consecutive_sell_block_windows = models.IntegerField()
    total_sell_block_windows = models.IntegerField()

    # Holder distribution metrics
    gini_coefficient = models.FloatField()
    total_holders = models.IntegerField()
    whale_count = models.IntegerField()
    whale_total_pct = models.FloatField()
    small_holders_pct = models.FloatField()
    holder_balance_std = models.FloatField()
    holder_balance_cv = models.FloatField()
    hhi_index = models.FloatField()

    # Advanced flags and scores
    inactive_token_flag = models.IntegerField()
    whale_domination_ratio = models.FloatField()
    whale_presence_flag = models.IntegerField()
    few_holders_flag = models.IntegerField()
    airdrop_like_flag = models.IntegerField()
    concentrated_large_community_score = models.FloatField()
    hhi_per_holder = models.FloatField()
    whale_but_no_small_flag = models.IntegerField()

    # Dynamic analyzer features (denormalized from HoneypotDaResult)
    balance_manipulation = models.IntegerField(default=0)
    buy_1 = models.IntegerField(default=0)
    buy_2 = models.IntegerField(default=0)
    buy_3 = models.IntegerField(default=0)
    existing_holders_check = models.IntegerField(default=0)
    exterior_call_check = models.IntegerField(default=0)
    sell_fail_type_1 = models.IntegerField(default=0)
    sell_fail_type_2 = models.IntegerField(default=0)
    sell_fail_type_3 = models.IntegerField(default=0)
    sell_result_1 = models.IntegerField(default=0)
    sell_result_2 = models.IntegerField(default=0)
    sell_result_3 = models.IntegerField(default=0)
    tax_manipulation = models.IntegerField(default=0)
    trading_suspend_check = models.IntegerField(default=0)
    unlimited_mint = models.IntegerField(default=0)

    class Meta:
        db_table = 'honeypot_processed_data'

    def __str__(self):
        return f"Honeypot data for token {self.token_info.id}"


class ExitProcessedDataInstance(models.Model):
    """
    Preprocessed data for exit scam detection analysis.
    Multiple records per token (one per event/transaction).
    """
    token_info = models.ForeignKey(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='exit_processed_instance',
        db_column='token_addr_idx'
    )

    event_time = models.DateTimeField(
        help_text="Event occurrence timestamp"
    )
    tx_hash = models.CharField(
        max_length=66,
        help_text="Transaction hash"
    )

    # Time delta
    delta_t_sec = models.FloatField(
        help_text="Time difference from previous event"
    )

    # Event type flags
    is_swap_event = models.IntegerField(
        help_text="1 if Swap event included, 0 otherwise"
    )

    # LP and reserves
    lp_total_supply = models.FloatField(
        null = True,
        blank = True,
        help_text="Current LP total supply"
    )
    reserve_base_drop_frac = models.FloatField(
        null = True,
        blank = True,
        help_text="Base token decrease rate"
    )
    reserve_quote = models.FloatField(
        help_text="Current major token balance"
    )
    reserve_quote_drop_frac = models.FloatField(
        null = True,
        blank = True,
        help_text="Major token decrease rate"
    )

    # Price and ratio
    price_ratio = models.FloatField(
        null = True,
        blank = True,
        help_text="Price ratio"
    )
    time_since_last_mint_sec = models.FloatField(
        null = True,
        blank = True,
        help_text="Time elapsed since last Mint (seconds)"
    )

    # LP and reserved 2
    lp_minted_amount_per_sec = models.FloatField(
        null = True,
        blank = True
    )
    lp_burned_amount_per_sec = models.FloatField(
        null = True,
        blank = True
    )
    recent_mint_ratio_last10 = models.FloatField(
        null = True,
        blank = True
    )
    recent_mint_ratio_last20 = models.FloatField(
        null = True,
        blank = True
    )
    recent_burn_ratio_last10 = models.FloatField(
        null = True,
        blank = True
    )
    recent_burn_ratio_last20 = models.FloatField(
        null = True,
        blank = True
    )
    reserve_quote_drawdown = models.FloatField(
        null = True,
        blank = True
    )

    # Mask Feature
    lp_total_supply_mask = models.FloatField(
        null = True
    )
    reserve_quote_mask = models.FloatField(
        null = True
    )
    price_ratio_mask = models.FloatField(
        null = True
    )
    time_since_last_mint_sec_mask = models.FloatField(
        null = True
    )

    class Meta:
        db_table = 'exit_processed_data_instance'
        ordering = ['token_info', 'event_time']
        indexes = [
            models.Index(fields=['token_info', 'event_time']),
            models.Index(fields=['tx_hash']),
        ]

    def __str__(self):
        return f"Exit data(instance) for token {self.token_info.id}, tx {self.tx_hash}"

class ExitProcessedDataStatic(models.Model):
    """
    Preprocessed data for exit scam detection analysis.
    Multiple records per token (one per event/transaction).
    """
    token_info = models.ForeignKey(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='exit_processed_static',
        db_column='token_addr_idx'
    )

    price_ratio_realized_vol = models.FloatField(
        null = True,
        blank = True
    )
    price_ratio_range  = models.FloatField(
        null = True,
        blank = True
    )
    reserve_quote_realized_vol = models.FloatField(
        null = True,
        blank = True
    )
    burn_ratio_all = models.FloatField(
        null = True,
        blank = True
    )
    reserve_quote_drawdown_global = models.FloatField(
        null = True,
        blank = True
    )
    swap_share = models.FloatField(
        null = True,
        blank = True
    )
    swaps_last5 = models.FloatField(
        null = True,
        blank = True
    )
    liquidity_age_days = models.FloatField(
        null = True,
        blank = True
    )
    holder_cnt = models.IntegerField(
        null = True,
        blank = True
    )


    
    class Meta:
        db_table = 'exit_processed_data_static'

    def __str__(self):
        return f"Exit data(static) for token {self.token_info.id}"

class HoneypotDaResult(models.Model):
    """
    Stores honeypot dynamic analysis results.
    One record per token, created by honeypot_DA module.
    """
    token_info = models.OneToOneField(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='honeypot_da_result',
        primary_key=True,
        db_column='token_addr_idx'
    )

    verified = models.BooleanField(
        help_text="Contract verified on Etherscan"
    )
    buy_1 = models.BooleanField()
    buy_2 = models.BooleanField()
    buy_3 = models.BooleanField()

    sell_1 = models.BooleanField()
    sell_2 = models.BooleanField()
    sell_3 = models.BooleanField()

    sell_fail_type_1 = models.IntegerField()
    sell_fail_type_2 = models.IntegerField()
    sell_fail_type_3 = models.IntegerField()

    # Trading suspend check
    trading_suspend_result = models.BooleanField()

    # Exterior call check
    exterior_call_result = models.BooleanField()

    # Unlimited mint
    unlimited_mint_result = models.BooleanField()

    # Balance manipulation
    balance_manipulation_result = models.BooleanField()

    # Tax manipulation
    tax_manipulation_result = models.BooleanField()

    # Existing holders check
    existing_holders_result = models.BooleanField()

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'honeypot_da_result'

    def __str__(self):
        return f"Honeypot DA result for token {self.token_info.id}"


class HoneypotMlResult(models.Model):
    token_info = models.OneToOneField(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='honeypot_ml_result',
        primary_key=True,
        db_column='token_addr_idx',
    )

    # y_pred (0/1) → is_honeypot
    is_honeypot = models.BooleanField(
        null=True,
        help_text="Honeypot prediction result (y_pred)"
    )

    # y_proba → probability
    probability = models.FloatField(
        null=True,
        help_text="Honeypot probability (y_proba, 0-1)"
    )

    # (선택) risk_level / threshold 는 그대로 둬도 되고, 나중에 계산 안 쓰면 null 허용으로 바꿔도 됨
    risk_level = models.CharField(
        max_length=20,
        help_text="Risk level: CRITICAL/HIGH/MEDIUM/LOW/VERY_LOW",
        null=True,
        blank=True,
    )
    threshold = models.FloatField(
        help_text="Decision threshold from model",
        null=True,
        blank=True,
    )

    # 🔄 기존 JSONField 제거
    # top_contributing_features = models.JSONField(
    #     help_text="Top 5 contributing features with values and directions"
    # )

    # ✅ 새로 추가: top1_feat ~ top5_feat, status
    top1_feat = models.CharField(
        max_length=64,
        null=True,
        blank=True,
        help_text="Most important feature name (rank 1)",
    )
    top1_feat_value = models.FloatField(
        null=True,
        blank=True,
        help_text="Value of top1 feature on this token",
    )

    top2_feat = models.CharField(
        max_length=64,
        null=True,
        blank=True,
        help_text="Feature name (rank 2)",
    )
    top2_feat_value = models.FloatField(
        null=True,
        blank=True,
        help_text="Value of top2 feature on this token",
    )

    top3_feat = models.CharField(
        max_length=64,
        null=True,
        blank=True,
        help_text="Feature name (rank 3)",
    )
    top3_feat_value = models.FloatField(
        null=True,
        blank=True,
        help_text="Value of top3 feature on this token",
    )

    top4_feat = models.CharField(
        max_length=64,
        null=True,
        blank=True,
        help_text="Feature name (rank 4)",
    )
    top4_feat_value = models.FloatField(
        null=True,
        blank=True,
        help_text="Value of top4 feature on this token",
    )

    top5_feat = models.CharField(
        max_length=64,
        null=True,
        blank=True,
        help_text="Feature name (rank 5)",
    )
    top5_feat_value = models.FloatField(
        null=True,
        blank=True,
        help_text="Value of top5 feature on this token",
    )

    status = models.CharField(
        max_length=20,
        null=True,
        default="PRED_ONLY",
        help_text="Result status (e.g., PRED_ONLY)",
    )

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'honeypot_ml_result'

    def __str__(self):
        return f"Honeypot ML result for token {self.token_info.id} (prob: {self.probability:.4f})"

class ExitMlResult(models.Model):
    """
    Stores exit scam ML analysis results.
    One record per token, created by exit_ML module.
    """
    token_info = models.OneToOneField(
        TokenInfo,
        on_delete=models.CASCADE,
        related_name='exit_ml_result',
        primary_key=True,
        db_column='token_addr_idx'
    )
    is_unformed_lp = models.IntegerField(
        default = 0,
        help_text="Is not Unformed-LP"
    )
    probability = models.FloatField(
        null = True,
        blank = True,
        help_text="Exit scam probability (0-1)"
    )
    tx_cnt = models.IntegerField(
        null = True,
        blank = True
    )
    timestamp = models.DateTimeField(
        null = True,
        blank = True
    )
    tx_hash = models.CharField(
        max_length = 66,
        null = True,
        blank = True
    )

    # Top Instance Feature
    reserve_base_drop_frac = models.FloatField(
        null = True,
        blank = True
    )
    reserve_quote = models.FloatField(
        null = True,
        blank = True
    )
    reserve_quote_drop_frac = models.FloatField(
        null = True,
        blank = True
    )
    price_ratio = models.FloatField(
        null = True,
        blank = True
    )
    time_since_last_mint_sec = models.FloatField(
        null = True,
        blank = True
    )
    
    # Static Feature
    liquidity_age_days = models.FloatField(
        null = True,
        blank = True
    )
    reserve_quote_drawdown_global = models.FloatField(
        null = True,
        blank = True
    )

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'exit_ml_result'

    def __str__(self):
        return f"Exit ML result for token {self.token_info.id} (prob: {self.probability:.4f})"