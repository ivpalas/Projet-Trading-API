"""
Training script for DQN Agent on GBP/USD trading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import json

from trading_env import TradingEnv
from dqn_agent import DQNAgent


class RLTrainer:
    """
    Trainer for Reinforcement Learning Trading Agent
    """
    
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        initial_balance: float = 10000.0,
        save_dir: str = "models/saved/rl"
    ):
        """
        Initialize RL Trainer
        
        Args:
            train_df: Training data with features
            val_df: Validation data (optional)
            initial_balance: Starting capital
            save_dir: Directory to save models
        """
        self.train_df = train_df
        self.val_df = val_df
        self.initial_balance = initial_balance
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environments
        self.train_env = TradingEnv(train_df, initial_balance=initial_balance)
        if val_df is not None:
            self.val_env = TradingEnv(val_df, initial_balance=initial_balance)
        
        # Initialize agent
        state_size = self.train_env.observation_space.shape[0]
        action_size = self.train_env.action_space.n
        
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update_frequency=10
        )
        
        # Training history
        self.train_history = {
            'episodes': [],
            'returns': [],
            'balances': [],
            'trades': [],
            'win_rates': [],
            'losses': [],
            'epsilons': []
        }
        
        self.val_history = {
            'episodes': [],
            'returns': [],
            'balances': [],
            'trades': [],
            'win_rates': []
        }
    
    def train_episode(self, episode: int) -> Dict:
        """
        Train agent for one episode
        
        Args:
            episode: Episode number
            
        Returns:
            Episode statistics
        """
        state, info = self.train_env.reset()
        total_reward = 0
        steps = 0
        losses = []
        
        while True:
            # Select action
            action = self.agent.select_action(state, training=True)
            
            # Execute action
            next_state, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent
            loss = self.agent.train_step()
            if loss > 0:
                losses.append(loss)
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # End episode
        self.agent.end_episode()
        
        # Get performance
        perf = self.train_env.get_performance_summary()
        
        return {
            'episode': episode,
            'total_reward': total_reward,
            'total_return': perf['total_return_pct'],
            'final_balance': perf['final_balance'],
            'total_trades': perf['total_trades'],
            'win_rate': perf['win_rate'],
            'steps': steps,
            'avg_loss': np.mean(losses) if losses else 0,
            'epsilon': self.agent.epsilon
        }
    
    def validate_episode(self, episode: int) -> Dict:
        """
        Validate agent on validation set
        
        Args:
            episode: Episode number
            
        Returns:
            Validation statistics
        """
        if self.val_df is None:
            return {}
        
        state, info = self.val_env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Select action (no exploration)
            action = self.agent.select_action(state, training=False)
            
            # Execute action
            next_state, reward, terminated, truncated, info = self.val_env.step(action)
            done = terminated or truncated
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Get performance
        perf = self.val_env.get_performance_summary()
        
        return {
            'episode': episode,
            'total_reward': total_reward,
            'total_return': perf['total_return_pct'],
            'final_balance': perf['final_balance'],
            'total_trades': perf['total_trades'],
            'win_rate': perf['win_rate'],
            'steps': steps
        }
    
    def train(
        self,
        num_episodes: int = 100,
        validate_every: int = 10,
        save_every: int = 50,
        verbose: bool = True
    ):
        """
        Train agent for multiple episodes
        
        Args:
            num_episodes: Number of training episodes
            validate_every: Validate every N episodes
            save_every: Save model every N episodes
            verbose: Print progress
        """
        print("=" * 80)
        print("TRAINING DQN AGENT")
        print("=" * 80)
        print(f"Training episodes: {num_episodes}")
        print(f"Validation: Every {validate_every} episodes")
        print(f"Device: {self.agent.device}")
        print()
        
        best_val_return = -np.inf
        
        for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
            # Train
            train_stats = self.train_episode(episode)
            
            # Store history
            self.train_history['episodes'].append(episode)
            self.train_history['returns'].append(train_stats['total_return'])
            self.train_history['balances'].append(train_stats['final_balance'])
            self.train_history['trades'].append(train_stats['total_trades'])
            self.train_history['win_rates'].append(train_stats['win_rate'])
            self.train_history['losses'].append(train_stats['avg_loss'])
            self.train_history['epsilons'].append(train_stats['epsilon'])
            
            # Validate
            if episode % validate_every == 0 and self.val_df is not None:
                val_stats = self.validate_episode(episode)
                
                self.val_history['episodes'].append(episode)
                self.val_history['returns'].append(val_stats['total_return'])
                self.val_history['balances'].append(val_stats['final_balance'])
                self.val_history['trades'].append(val_stats['total_trades'])
                self.val_history['win_rates'].append(val_stats['win_rate'])
                
                # Save best model
                if val_stats['total_return'] > best_val_return:
                    best_val_return = val_stats['total_return']
                    self.save_agent(f"best_agent_ep{episode}.pth")
                
                if verbose:
                    print(f"\nEpisode {episode}")
                    print(f"  Train - Return: {train_stats['total_return']:.2f}%, "
                          f"Balance: {train_stats['final_balance']:.2f}, "
                          f"Trades: {train_stats['total_trades']}, "
                          f"Win Rate: {train_stats['win_rate']:.2%}")
                    print(f"  Val   - Return: {val_stats['total_return']:.2f}%, "
                          f"Balance: {val_stats['final_balance']:.2f}, "
                          f"Trades: {val_stats['total_trades']}, "
                          f"Win Rate: {val_stats['win_rate']:.2%}")
                    print(f"  Epsilon: {train_stats['epsilon']:.4f}, "
                          f"Loss: {train_stats['avg_loss']:.4f}")
            
            # Save periodically
            if episode % save_every == 0:
                self.save_agent(f"agent_ep{episode}.pth")
                self.save_history()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        
        # Save final model
        self.save_agent("final_agent.pth")
        self.save_history()
        
        # Print summary
        self.print_summary()
    
    def save_agent(self, filename: str):
        """Save agent to file"""
        filepath = self.save_dir / filename
        self.agent.save(str(filepath))
    
    def save_history(self):
        """Save training history"""
        history_file = self.save_dir / "training_history.json"
        
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"History saved to {history_file}")
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        
        # Training stats
        print("\nTraining:")
        print(f"  Episodes: {len(self.train_history['episodes'])}")
        print(f"  Avg Return: {np.mean(self.train_history['returns']):.2f}%")
        print(f"  Max Return: {np.max(self.train_history['returns']):.2f}%")
        print(f"  Final Return: {self.train_history['returns'][-1]:.2f}%")
        print(f"  Avg Trades: {np.mean(self.train_history['trades']):.0f}")
        print(f"  Avg Win Rate: {np.mean(self.train_history['win_rates']):.2%}")
        
        # Validation stats
        if self.val_history['episodes']:
            print("\nValidation:")
            print(f"  Episodes: {len(self.val_history['episodes'])}")
            print(f"  Avg Return: {np.mean(self.val_history['returns']):.2f}%")
            print(f"  Max Return: {np.max(self.val_history['returns']):.2f}%")
            print(f"  Final Return: {self.val_history['returns'][-1]:.2f}%")
            print(f"  Avg Trades: {np.mean(self.val_history['trades']):.0f}")
            print(f"  Avg Win Rate: {np.mean(self.val_history['win_rates']):.2%}")
    
    def plot_training_progress(self, save_fig: bool = True):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Returns
        axes[0, 0].plot(self.train_history['episodes'], 
                       self.train_history['returns'], 
                       label='Train', alpha=0.7)
        if self.val_history['episodes']:
            axes[0, 0].plot(self.val_history['episodes'], 
                           self.val_history['returns'], 
                           label='Validation', alpha=0.7)
        axes[0, 0].set_title('Total Return (%)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Return %')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Balance
        axes[0, 1].plot(self.train_history['episodes'], 
                       self.train_history['balances'], 
                       label='Train', alpha=0.7)
        if self.val_history['episodes']:
            axes[0, 1].plot(self.val_history['episodes'], 
                           self.val_history['balances'], 
                           label='Validation', alpha=0.7)
        axes[0, 1].set_title('Final Balance')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Balance ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=self.initial_balance, color='black', linestyle='--', alpha=0.3)
        
        # Epsilon and Loss
        axes[1, 0].plot(self.train_history['episodes'], 
                       self.train_history['epsilons'], 
                       label='Epsilon', color='orange')
        axes[1, 0].set_title('Exploration Rate (Epsilon)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Win Rate
        axes[1, 1].plot(self.train_history['episodes'], 
                       self.train_history['win_rates'], 
                       label='Train', alpha=0.7)
        if self.val_history['episodes']:
            axes[1, 1].plot(self.val_history['episodes'], 
                           self.val_history['win_rates'], 
                           label='Validation', alpha=0.7)
        axes[1, 1].set_title('Win Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            fig_path = self.save_dir / "training_progress.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {fig_path}")
        
        plt.show()


def main():
    """Main training script"""
    
    # Load data
    print("Loading data...")
    df_2022 = pd.read_parquet('data/processed/ml_dataset_2022.parquet')
    df_2023 = pd.read_parquet('data/processed/ml_dataset_2023.parquet')
    
    print(f"Train data (2022): {len(df_2022)} rows")
    print(f"Val data (2023): {len(df_2023)} rows")
    
    # Initialize trainer
    trainer = RLTrainer(
        train_df=df_2022,
        val_df=df_2023,
        initial_balance=10000.0,
        save_dir="models/saved/rl"
    )
    
    # Train agent
    trainer.train(
        num_episodes=20,
        validate_every=5,
        save_every=10,
        verbose=True
    )
    
    # Plot results
    trainer.plot_training_progress(save_fig=True)


if __name__ == "__main__":
    main()