"""
Custom Gym Environment for GBP/USD Trading
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional


class TradingEnv(gym.Env):
    """
    Custom Trading Environment for Reinforcement Learning
    
    State Space:
        - Price features (OHLC, returns, etc.)
        - Technical indicators (RSI, MACD, EMAs, etc.)
        - Position info (current position, unrealized P&L)
        
    Action Space:
        - 0: HOLD (do nothing)
        - 1: BUY (go long)
        - 2: SELL (go short / close long)
        
    Reward:
        - Profit/Loss from trades
        - Penalty for holding losing positions
        - Bonus for winning trades
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.0001,  # 1 pip spread
        max_position: float = 1.0,
        lookback_window: int = 20
    ):
        """
        Initialize Trading Environment
        
        Args:
            df: DataFrame with OHLC data and features
            initial_balance: Starting capital
            transaction_cost: Cost per trade (spread)
            max_position: Maximum position size (fraction of balance)
            lookback_window: Number of past candles to include in state
        """
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.lookback_window = lookback_window
        
        # Identify feature columns (exclude OHLC, target, and datetime)
        self.price_cols = ['open_15m', 'high_15m', 'low_15m', 'close_15m']
        exclude_cols = self.price_cols + ['target', 'volume_15m', 'timestamp', 'timestamp_15m']
        
        # Get feature columns and exclude datetime types
        self.feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    self.feature_cols.append(col)
        
        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        
        # Observation space: features + position info
        n_features = len(self.feature_cols) + 3  # +3 for position, unrealized_pnl, balance
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.done = False
        
        # Trading state
        self.balance = initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.entry_price = 0
        self.unrealized_pnl = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # History
        self.balance_history = []
        self.action_history = []
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.done = False
        
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.unrealized_pnl = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        self.balance_history = [self.balance]
        self.action_history = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        # Get current price
        current_price = self.df.loc[self.current_step, 'close_15m']
        
        # Execute action and calculate reward
        reward = self._execute_action(action, current_price)
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = self.balance <= 0  # Bankrupt
        self.done = terminated or truncated
        
        # Update history
        self.balance_history.append(self.balance)
        self.action_history.append(action)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """
        Execute trading action and return reward
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            current_price: Current market price
            
        Returns:
            reward: Reward for this action
        """
        reward = 0
        
        # Close existing position if any
        if self.position != 0:
            pnl = self._calculate_pnl(current_price)
            self.balance += pnl
            
            # Reward based on profit/loss
            reward += pnl
            
            # Track winning trades
            if pnl > 0:
                self.winning_trades += 1
            
            self.position = 0
            self.entry_price = 0
            self.unrealized_pnl = 0
        
        # Execute new action
        if action == 1:  # BUY (go long)
            position_size = self.balance * self.max_position
            cost = position_size * self.transaction_cost
            
            self.position = 1
            self.entry_price = current_price
            self.balance -= cost
            self.total_trades += 1
            
            # Small penalty for transaction cost
            reward -= cost
            
        elif action == 2:  # SELL (go short)
            position_size = self.balance * self.max_position
            cost = position_size * self.transaction_cost
            
            self.position = -1
            self.entry_price = current_price
            self.balance -= cost
            self.total_trades += 1
            
            # Small penalty for transaction cost
            reward -= cost
        
        # HOLD (action == 0): do nothing
        
        # Update unrealized P&L if in position
        if self.position != 0:
            self.unrealized_pnl = self._calculate_pnl(current_price)
            
            # Small reward/penalty for unrealized P&L
            reward += self.unrealized_pnl * 0.1
        
        return reward
    
    def _calculate_pnl(self, current_price: float) -> float:
        """Calculate profit/loss for current position"""
        if self.position == 0:
            return 0
        
        position_size = self.balance * self.max_position
        price_change = current_price - self.entry_price
        
        if self.position == 1:  # Long
            pnl = (price_change / self.entry_price) * position_size
        else:  # Short
            pnl = -(price_change / self.entry_price) * position_size
        
        return pnl
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        # Get current features
        features = self.df.loc[self.current_step, self.feature_cols].values
        
        # Add position info
        position_info = np.array([
            self.position,  # Current position (-1, 0, 1)
            self.unrealized_pnl / self.initial_balance,  # Normalized unrealized P&L
            self.balance / self.initial_balance  # Normalized balance
        ])
        
        # Combine features
        obs = np.concatenate([features, position_info]).astype(np.float32)
        
        # Handle NaN values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades)
        }
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            info = self._get_info()
            print(f"Step: {info['step']}")
            print(f"Balance: {info['balance']:.2f}")
            print(f"Position: {info['position']}")
            print(f"Unrealized P&L: {info['unrealized_pnl']:.2f}")
            print(f"Total Trades: {info['total_trades']}")
            print(f"Win Rate: {info['win_rate']:.2%}")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary at end of episode"""
        total_return = (self.balance / self.initial_balance - 1) * 100
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return_pct': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.total_trades - self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_steps': self.current_step,
            'balance_history': self.balance_history,
            'action_history': self.action_history
        }