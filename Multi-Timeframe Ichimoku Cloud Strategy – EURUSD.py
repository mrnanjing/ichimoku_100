
# ============================================================
#  Multi-Timeframe Ichimoku Cloud Strategy – EURUSD
#  Platform : QuantConnect (QCAlgorithm)
#  Primary  : 1-Hour bars   (QuoteBar – Forex)
#  Secondary: 4-Hour bars   (confluence filter)
#
#  STRICT SINGLE-TRADE EDITION – August 2026
#  ---------------------------
#  FIX A: Enforce ONLY ONE active trade at a time (no state stomping)
#  FIX B: Sync state with actual Portfolio.Quantity every bar
#  FIX C: Tighter MAX_STOP_PIPS = 20 pips (was 50)
#  FIX D: Tighter risk 0.25% / trade, 0.5% max trade DD
#  FIX E: Max hold 20 bars (20 hours) – was 30
#  FIX F: Add is_entering flag to prevent double-entry on same bar
#  FIX G: Orphaned-state auto-recovery if Quantity==0 but flags say open
# ============================================================

from AlgorithmImports import *
from collections import deque
import numpy as np


class MultiTimeframeIchimokuStrategy(QCAlgorithm):

    # ----------------------------------------------------------
    # Ichimoku parameters
    # ----------------------------------------------------------
    TENKAN      = 9
    KIJUN       = 26
    SENKOU_B    = 52
    CLOUD_DELAY = 26

    # ----------------------------------------------------------
    # FIXED Risk Management
    # ----------------------------------------------------------
    RISK_PER_TRADE      = 0.0025   ### FIX D: 0.25% equity per trade (was 0.5%)
    ATR_PERIOD          = 14
    ATR_STOP_MULT       = 1.5
    ATR_TRAIL_MULT      = 1.0
    MIN_STOP_PIPS       = 0.0008   ### FIX C: 8-pip floor (was 5)
    MAX_STOP_PIPS       = 0.0020   ### FIX C: 20-pip hard ceiling (was 50!)
    ATR_4H_MULT         = 1.0

    MAX_HOLD_BARS       = 20       ### FIX E: 20 hours max (was 30)
    MAX_TRADE_DD_PCT    = 0.005    ### FIX D: 0.5% hard cap per trade (was 1.0%)

    MAX_DAILY_DD        = 0.03
    MAX_WEEKLY_LOSS     = 0.05
    MAX_CONSEC_LOSS     = 5
    RISK_REDUCTION      = 0.50

    SCALE1_R            = 1.5

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2026, 8, 19)
        self.SetCash(100_000)

        self.symbol = self.AddForex(
            "EURUSD",
            Resolution.Hour,
            Market.Oanda
        ).Symbol

        # Consolidators
        self.bar1h = QuoteBarConsolidator(timedelta(hours=1))
        self.bar1h.DataConsolidated += self.on_1h_bar
        self.SubscriptionManager.AddConsolidator(self.symbol, self.bar1h)

        self.bar4h = QuoteBarConsolidator(timedelta(hours=4))
        self.bar4h.DataConsolidated += self.on_4h_bar
        self.SubscriptionManager.AddConsolidator(self.symbol, self.bar4h)

        # ATR (1H native)
        self.atr = self.ATR(self.symbol, self.ATR_PERIOD,
                            MovingAverageType.Wilders, Resolution.Hour)

        # Rolling windows
        WINDOW = 200
        self.closes_1h = deque(maxlen=WINDOW)
        self.highs_1h  = deque(maxlen=WINDOW)
        self.lows_1h   = deque(maxlen=WINDOW)
        self.closes_4h = deque(maxlen=WINDOW)
        self.highs_4h  = deque(maxlen=WINDOW)
        self.lows_4h   = deque(maxlen=WINDOW)

        # State tracking
        self.position       = 0
        self.entry_price    = None
        self.stop_price     = None
        self.prev_tenkan    = None
        self.prev_kijun     = None

        # Trade log
        self.trade_log  = []
        self.entry_time = None

        # Risk-management state
        self.daily_equity_high   = 100_000
        self.weekly_start_equity = 100_000
        self.consecutive_losses  = 0
        self.trading_paused      = False
        self._last_equity_date   = None
        self._last_week_date     = None

        # Order tickets
        self.stop_ticket    = None
        self.tp1_ticket     = None
        self.entry_ticket   = None   ### FIX F: track entry ticket

        # Phase tracking
        self.phase          = 0
        self.full_quantity  = 0
        self.runner_qty     = 0
        self.risk_unit      = 0.0

        # Safety tracking
        self.bars_in_trade  = 0
        self.is_exiting     = False
        self.is_entering    = False  ### FIX F: prevent double-entry
        self.max_trade_dd   = 0.0

        self.SetWarmUp(timedelta(days=120))

        self.Log("Strategy initialised – STRICT SINGLE-TRADE Ichimoku EURUSD")

    # ==========================================================
    #  RISK GUARDS
    # ==========================================================
    def _update_risk_state(self):
        equity = self.Portfolio.TotalPortfolioValue
        today  = self.Time.date()
        week   = self.Time.isocalendar()[1]

        if self._last_equity_date != today:
            self._last_equity_date = today
            self.daily_equity_high = equity
        else:
            self.daily_equity_high = max(self.daily_equity_high, equity)

        if self._last_week_date != week:
            self._last_week_date = week
            self.weekly_start_equity = equity

        daily_dd    = (self.daily_equity_high - equity) / self.daily_equity_high
        weekly_loss = (self.weekly_start_equity - equity) / self.weekly_start_equity

        if daily_dd > self.MAX_DAILY_DD or weekly_loss > self.MAX_WEEKLY_LOSS:
            self.trading_paused = True
            self.Log(f"RISK PAUSE | Daily DD {daily_dd:.2%} | Weekly Loss {weekly_loss:.2%}")
        else:
            if self.trading_paused and daily_dd <= self.MAX_DAILY_DD * 0.8:
                self.trading_paused = False

    def _risk_multiplier(self):
        if self.consecutive_losses >= self.MAX_CONSEC_LOSS:
            return self.RISK_REDUCTION
        return 1.0

    # ==========================================================
    #  ICHIMOKU CALCULATION HELPERS
    # ==========================================================
    @staticmethod
    def _donchian_mid(highs_list, lows_list, period):
        h = max(highs_list[:period])
        l = min(lows_list[:period])
        return (h + l) / 2.0

    def _compute_ichimoku(self, closes, highs, lows):
        n = len(closes)
        min_bars = self.SENKOU_B + self.CLOUD_DELAY
        if n < min_bars:
            return None

        h_list = list(highs)
        l_list = list(lows)
        c_list = list(closes)

        tenkan = self._donchian_mid(h_list, l_list, self.TENKAN)
        kijun  = self._donchian_mid(h_list, l_list, self.KIJUN)

        ph = h_list[self.CLOUD_DELAY:]
        pl = l_list[self.CLOUD_DELAY:]

        t_past   = self._donchian_mid(ph, pl, self.TENKAN)
        k_past   = self._donchian_mid(ph, pl, self.KIJUN)
        senkou_a = (t_past + k_past) / 2.0
        senkou_b = self._donchian_mid(ph, pl, self.SENKOU_B)

        cloud_top    = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        close_now    = c_list[0]
        close_26_ago = c_list[self.CLOUD_DELAY] if n > self.CLOUD_DELAY else None

        return dict(
            tenkan       = tenkan,
            kijun        = kijun,
            senkou_a     = senkou_a,
            senkou_b     = senkou_b,
            cloud_top    = cloud_top,
            cloud_bottom = cloud_bottom,
            close        = close_now,
            close_26_ago = close_26_ago,
        )

    # ==========================================================
    #  4-HOUR BAR HANDLER
    # ==========================================================
    def on_4h_bar(self, sender, bar):
        mid_close = (bar.Bid.Close + bar.Ask.Close) / 2.0
        mid_high  = (bar.Bid.High  + bar.Ask.High)  / 2.0
        mid_low   = (bar.Bid.Low   + bar.Ask.Low)   / 2.0

        self.closes_4h.appendleft(mid_close)
        self.highs_4h .appendleft(mid_high)
        self.lows_4h  .appendleft(mid_low)

    def _4h_cloud_direction(self):
        n = len(self.closes_4h)
        if n < self.SENKOU_B + self.CLOUD_DELAY:
            return 0

        ph = list(self.highs_4h)[self.CLOUD_DELAY:]
        pl = list(self.lows_4h) [self.CLOUD_DELAY:]

        t_past   = self._donchian_mid(ph, pl, self.TENKAN)
        k_past   = self._donchian_mid(ph, pl, self.KIJUN)
        senkou_a = (t_past + k_past) / 2.0
        senkou_b = self._donchian_mid(ph, pl, self.SENKOU_B)

        if   senkou_a > senkou_b: return  1
        elif senkou_a < senkou_b: return -1
        return 0

    # ==========================================================
    #  1-HOUR BAR HANDLER
    # ==========================================================
    def on_1h_bar(self, sender, bar):
        mid_close = (bar.Bid.Close + bar.Ask.Close) / 2.0
        mid_high  = (bar.Bid.High  + bar.Ask.High)  / 2.0
        mid_low   = (bar.Bid.Low   + bar.Ask.Low)   / 2.0

        self.closes_1h.appendleft(mid_close)
        self.highs_1h .appendleft(mid_high)
        self.lows_1h  .appendleft(mid_low)

        if self.IsWarmingUp:
            return

        # ── FIX G: Orphaned-state recovery ─────────────────────
        # If broker shows flat but our flags say we have a trade,
        # some previous exit was missed. Hard-reset immediately.
        actual_qty = self.Portfolio[self.symbol].Quantity
        if self.position != 0 and actual_qty == 0 and not self.is_exiting:
            self.Log("ORPHAN DETECTED: Portfolio flat but state still open. Hard-resetting.")
            self._hard_reset_state()
            return

        self._update_risk_state()

        ichi = self._compute_ichimoku(self.closes_1h, self.highs_1h, self.lows_1h)
        if ichi is None:
            return

        tenkan       = ichi["tenkan"]
        kijun        = ichi["kijun"]
        senkou_a     = ichi["senkou_a"]
        senkou_b     = ichi["senkou_b"]
        cloud_top    = ichi["cloud_top"]
        cloud_bottom = ichi["cloud_bottom"]
        close        = ichi["close"]
        close_26_ago = ichi["close_26_ago"]

        cloud_4h_dir = self._4h_cloud_direction()
        cloud_1h_dir = (1 if senkou_a > senkou_b else
                       (-1 if senkou_a < senkou_b else 0))

        # ── EXIT logic ─────────────────────────────────────────
        if self.position != 0 and not self.is_exiting:
            self.bars_in_trade += 1

            self._check_emergency_exit(close, bar.EndTime)
            self._check_time_exit(bar.EndTime)

            if not self.is_exiting:
                self._check_exit(
                    close, tenkan, kijun,
                    senkou_a, senkou_b, cloud_top, cloud_bottom,
                    close_26_ago, bar.EndTime
                )

        # ── ENTRY logic ────────────────────────────────────────
        ### FIX A: Only enter if completely flat AND not already entering/exiting
        if (self.position == 0 and
            actual_qty == 0 and
            not self.is_entering and
            not self.is_exiting and
            self.prev_tenkan is not None):

            self._check_entry(
                close, tenkan, kijun,
                senkou_a, senkou_b, cloud_top, cloud_bottom,
                close_26_ago, cloud_4h_dir, cloud_1h_dir, bar.EndTime
            )

        self.prev_tenkan = tenkan
        self.prev_kijun  = kijun

    # ==========================================================
    #  EMERGENCY DRAWDOWN EXIT
    # ==========================================================
    def _check_emergency_exit(self, close, bar_time):
        if self.position == 0 or self.entry_price is None:
            return

        if self.position == 1:
            unrealized = (close - self.entry_price) * self.full_quantity
        else:
            unrealized = (self.entry_price - close) * self.full_quantity

        self.max_trade_dd = min(self.max_trade_dd, unrealized)

        equity = self.Portfolio.TotalPortfolioValue
        max_allowed_dd = equity * self.MAX_TRADE_DD_PCT

        if unrealized < -max_allowed_dd:
            self._force_exit(close, bar_time,
                f"EMERGENCY: Unrealized loss ${abs(unrealized):.0f} > {self.MAX_TRADE_DD_PCT:.1%} equity")

    # ==========================================================
    #  TIME-BASED EXIT
    # ==========================================================
    def _check_time_exit(self, bar_time):
        if self.bars_in_trade >= self.MAX_HOLD_BARS:
            self._force_exit(self.entry_price, bar_time,
                f"TIME EXIT: Held {self.bars_in_trade} bars (max {self.MAX_HOLD_BARS})")

    # ==========================================================
    #  ENTRY EVALUATION
    # ==========================================================
    def _check_entry(self, close, tenkan, kijun,
                     senkou_a, senkou_b, cloud_top, cloud_bottom,
                     close_26_ago, cloud_4h_dir, cloud_1h_dir, bar_time):

        if close_26_ago is None:
            return
        if self.trading_paused:
            return
        if not self.atr.IsReady:
            return

        long_signal = (
            cloud_4h_dir == 1 and cloud_1h_dir == 1
            and senkou_a > senkou_b
            and close > cloud_top
            and self.prev_tenkan <= self.prev_kijun
            and tenkan > kijun
            and close > close_26_ago
        )

        short_signal = (
            cloud_4h_dir == -1 and cloud_1h_dir == -1
            and senkou_a < senkou_b
            and close < cloud_bottom
            and self.prev_tenkan >= self.prev_kijun
            and tenkan < kijun
            and close < close_26_ago
        )

        if long_signal:
            self._enter_trade(1, close, kijun, bar_time)
        elif short_signal:
            self._enter_trade(-1, close, kijun, bar_time)

    # ==========================================================
    #  FIXED POSITION SIZING & HARD ORDER ENTRY
    # ==========================================================
    def _enter_trade(self, direction, entry_price, kijun, bar_time):
        ### FIX F: Lock entry immediately so no other bar can enter
        self.is_entering = True

        equity = self.Portfolio.TotalPortfolioValue
        atr_val = self.atr.Current.Value if self.atr.IsReady else 0.0010

        atr_1h   = atr_val * self.ATR_4H_MULT
        atr_dist = atr_1h * self.ATR_STOP_MULT
        stop_dist = min(max(atr_dist, self.MIN_STOP_PIPS), self.MAX_STOP_PIPS)

        if direction == 1:
            stop_price = entry_price - stop_dist
        else:
            stop_price = entry_price + stop_dist

        risk_mult    = self._risk_multiplier()
        risk_dollars = equity * self.RISK_PER_TRADE * risk_mult

        raw_qty  = risk_dollars / stop_dist
        quantity = max(1_000, round(raw_qty / 1_000) * 1_000)

        while quantity * stop_dist > risk_dollars * 1.05 and quantity > 1_000:
            quantity -= 1_000

        half_qty   = max(1_000, round(quantity / 2 / 1_000) * 1_000)
        runner_qty = quantity - half_qty

        self.risk_unit = stop_dist
        tp1_dist = self.risk_unit * self.SCALE1_R
        if direction == 1:
            tp1_price = entry_price + tp1_dist
        else:
            tp1_price = entry_price - tp1_dist

        signed_qty = quantity if direction == 1 else -quantity

        # ── Submit orders ─────────────────────────────────────
        self.entry_ticket = self.MarketOrder(self.symbol, signed_qty)

        if direction == 1:
            self.stop_ticket = self.StopMarketOrder(self.symbol, -quantity, stop_price)
        else:
            self.stop_ticket = self.StopMarketOrder(self.symbol,  quantity, stop_price)

        if direction == 1:
            self.tp1_ticket = self.LimitOrder(self.symbol, -half_qty, tp1_price)
        else:
            self.tp1_ticket = self.LimitOrder(self.symbol,  half_qty, tp1_price)

        # ── Record state ──────────────────────────────────────
        self.position      = direction
        self.entry_price   = entry_price
        self.stop_price    = stop_price
        self.phase         = 1
        self.full_quantity = quantity
        self.runner_qty    = runner_qty
        self.entry_time    = bar_time
        self.bars_in_trade = 0
        self.is_exiting    = False
        self.max_trade_dd  = 0.0

        self.Log(
            f"ENTER {'LONG' if direction==1 else 'SHORT'} | "
            f"Price={entry_price:.5f} | SL={stop_price:.5f} "
            f"({stop_dist*10000:.1f} pips) | "
            f"TP1={tp1_price:.5f} ({self.SCALE1_R}R) | "
            f"Qty={quantity:,} | Risk=${quantity*stop_dist:.2f} | "
            f"RiskMult={risk_mult:.0%} | 1H_ATR={atr_1h*10000:.1f}p"
        )

    # ==========================================================
    #  ORDER EVENTS
    # ==========================================================
    def OnOrderEvent(self, order_event):
        if order_event.Status != OrderStatus.Filled:
            return

        if self.is_exiting:
            return

        # Entry fill confirmation
        if (self.entry_ticket is not None
                and order_event.OrderId == self.entry_ticket.OrderId):
            ### FIX F: Entry filled – clear entering lock
            self.is_entering = False
            return

        # Phase-1 TP hit → move to Phase-2
        if (self.tp1_ticket is not None
                and order_event.OrderId == self.tp1_ticket.OrderId
                and self.phase == 1):

            self.phase = 2
            self.Log(
                f"PHASE-1 TP HIT | Runner={self.runner_qty:,} | "
                f"Moving stop to break-even {self.entry_price:.5f}"
            )

            if self.stop_ticket is not None:
                self.stop_ticket.Cancel()

            be = self.entry_price
            if self.position == 1:
                self.stop_ticket = self.StopMarketOrder(
                    self.symbol, -self.runner_qty, be
                )
            else:
                self.stop_ticket = self.StopMarketOrder(
                    self.symbol,  self.runner_qty, be
                )
            self.stop_price = be
            return

        # Hard stop filled → record & reset
        if (self.stop_ticket is not None
              and order_event.OrderId == self.stop_ticket.OrderId):

            self.is_exiting = True
            self._record_and_reset(order_event.FillPrice, "Hard Stop / Break-even")
            return

    # ==========================================================
    #  EXIT EVALUATION
    # ==========================================================
    def _check_exit(self, close, tenkan, kijun,
                    senkou_a, senkou_b, cloud_top, cloud_bottom,
                    close_26_ago, bar_time):

        if close_26_ago is None or self.position == 0 or self.is_exiting:
            return

        reason = None

        if self.position == 1:
            if senkou_a <= senkou_b:
                reason = "Rule 1 broken – Cloud turned RED"
            elif close <= cloud_top:
                reason = "Rule 2 broken – Price entered/below cloud"
            elif tenkan <= kijun:
                reason = "Rule 3 broken – Tenkan crossed below Kijun"
            elif close <= close_26_ago:
                reason = "Rule 4 broken – Chikou below price[26]"

            if self.phase == 2 and reason is None:
                atr_val = self.atr.Current.Value if self.atr.IsReady else 0.0
                trail = kijun - (atr_val * self.ATR_TRAIL_MULT)
                if trail > self.stop_price:
                    self._update_stop(trail)

        elif self.position == -1:
            if senkou_a >= senkou_b:
                reason = "Rule 1 broken – Cloud turned GREEN"
            elif close >= cloud_bottom:
                reason = "Rule 2 broken – Price entered/above cloud"
            elif tenkan >= kijun:
                reason = "Rule 3 broken – Tenkan crossed above Kijun"
            elif close >= close_26_ago:
                reason = "Rule 4 broken – Chikou above price[26]"

            if self.phase == 2 and reason is None:
                atr_val = self.atr.Current.Value if self.atr.IsReady else 0.0
                trail = kijun + (atr_val * self.ATR_TRAIL_MULT)
                if trail < self.stop_price:
                    self._update_stop(trail)

        if reason:
            self.is_exiting = True
            self._force_exit(close, bar_time, reason)

    # ==========================================================
    #  HELPERS
    # ==========================================================
    def _update_stop(self, new_stop):
        if self.stop_ticket is not None:
            self.stop_ticket.Cancel()

        qty = self.runner_qty if self.phase == 2 else self.full_quantity
        if self.position == 1:
            self.stop_ticket = self.StopMarketOrder(self.symbol, -qty, new_stop)
        else:
            self.stop_ticket = self.StopMarketOrder(self.symbol,  qty, new_stop)
        self.stop_price = new_stop

    def _force_exit(self, close, bar_time, reason):
        if self.is_exiting and self.position == 0:
            return

        self.is_exiting = True

        for ticket in [self.stop_ticket, self.tp1_ticket]:
            if ticket is not None:
                try:
                    ticket.Cancel()
                except Exception:
                    pass

        self.Liquidate(self.symbol)
        self._record_and_reset(close, reason)

    def _record_and_reset(self, exit_price, reason):
        pnl_pips = (exit_price - self.entry_price) * self.position * 10_000
        duration_hrs = (
            (self.Time - self.entry_time).total_seconds() / 3_600
            if self.entry_time else 0
        )

        if pnl_pips <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self.trade_log.append({
            "direction"    : self.position,
            "phase_at_exit": self.phase,
            "entry_time"   : self.entry_time,
            "exit_time"    : self.Time,
            "entry_price"  : self.entry_price,
            "exit_price"   : exit_price,
            "pnl_pips"     : pnl_pips,
            "duration_hrs" : duration_hrs,
            "reason"       : reason,
            "max_dd"       : self.max_trade_dd,
        })

        self.Log(
            f"EXIT {'LONG' if self.position==1 else 'SHORT'} "
            f"(Phase {self.phase}) | "
            f"Price={exit_price:.5f} | PnL={pnl_pips:+.1f} pips | "
            f"Dur={duration_hrs:.1f}h | MaxDD={self.max_trade_dd:+.2f} | Reason: {reason} | "
            f"LossStreak={self.consecutive_losses}"
        )

        self._hard_reset_state()

    def _hard_reset_state(self):
        """Atomic reset of ALL trade state. Called on normal exit AND orphan recovery."""
        self.position      = 0
        self.entry_price   = None
        self.stop_price    = None
        self.phase         = 0
        self.full_quantity = 0
        self.runner_qty    = 0
        self.stop_ticket   = None
        self.tp1_ticket    = None
        self.entry_ticket  = None
        self.entry_time    = None
        self.risk_unit     = 0.0
        self.bars_in_trade = 0
        self.is_exiting    = False
        self.is_entering   = False   ### FIX F: clear entry lock
        self.max_trade_dd  = 0.0

    # ==========================================================
    #  END-OF-ALGORITHM SUMMARY
    # ==========================================================
    def OnEndOfAlgorithm(self):
        trades = self.trade_log
        total  = len(trades)

        if total == 0:
            self.Log("No trades executed during back-test.")
            return

        wins     = [t for t in trades if t["pnl_pips"] > 0]
        losses   = [t for t in trades if t["pnl_pips"] <= 0]
        win_rate = len(wins) / total * 100

        avg_dur      = np.mean([t["duration_hrs"] for t in trades])
        avg_win_pips = np.mean([t["pnl_pips"] for t in wins])  if wins   else 0
        avg_los_pips = np.mean([t["pnl_pips"] for t in losses]) if losses else 0

        equity_final = self.Portfolio.TotalPortfolioValue
        total_return = (equity_final / 100_000 - 1) * 100

        reason_counts = {}
        for t in trades:
            reason_counts[t["reason"]] = reason_counts.get(t["reason"], 0) + 1

        self.Log("=" * 60)
        self.Log("  MULTI-TF ICHIMOKU – PERFORMANCE SUMMARY (STRICT SINGLE-TRADE)")
        self.Log("=" * 60)
        self.Log(f"  Total Trades   : {total}")
        self.Log(f"  Win Rate       : {win_rate:.1f}%  "
                 f"({len(wins)} wins / {len(losses)} losses)")
        self.Log(f"  Total Return   : {total_return:.2f}%")
        self.Log(f"  Final Equity   : ${equity_final:,.2f}")
        self.Log(f"  Avg Duration   : {avg_dur:.1f} hours")
        self.Log(f"  Avg Win (pips) : {avg_win_pips:.1f}")
        self.Log(f"  Avg Loss (pips): {avg_los_pips:.1f}")
        self.Log(f"  Best Trade     : {max(t['pnl_pips'] for t in trades):.1f} pips")
        self.Log(f"  Worst Trade    : {min(t['pnl_pips'] for t in trades):.1f} pips")
        self.Log("  Exit reasons:")
        for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
            self.Log(f"    {r:<45}: {c}")
        self.Log("=" * 60)

        self.Plot("Summary", "Win Rate %",       win_rate)
        self.Plot("Summary", "Total Trades",     total)
        self.Plot("Summary", "Avg Duration hrs", avg_dur)
        self.Plot("Summary", "Total Return %",   total_return)
