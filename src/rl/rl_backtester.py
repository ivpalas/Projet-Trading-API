"""
Backtester for trained RL agents
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import torch

from trading_env import TradingEnv
from dqn_agent import DQNAgent


class RLBacktester:
    """
    Backtester for trained RL trading agents
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize RL Backtester
        
        Args:
            initial_balance: Starting capital for backtesting
        """
        self.initial_balance = initial_balance
        self.results = {}
    
    def backtest_agent(
        self,
        agent: DQNAgent,
        test_df: pd.DataFrame,
        agent_name: str = "DQN Agent"
    ) -> Dict:
        """
        Backtest a trained agent on test data
        
        Args:
            agent: Trained DQN agent
            test_df: Test data with features
            agent_name: Name for the agent
            
        Returns:
            Backtest results dictionary
        """
        print(f"\n{'='*80}")
        print(f"Backtesting {agent_name}")
        print(f"{'='*80}")
        
        # Create test environment
        env = TradingEnv(test_df, initial_balance=self.initial_balance)
        
        # Run episode
        state, info = env.reset()
        actions_taken = []
        rewards_received = []
        balance_history = []
        
        steps = 0
        
        while True:
            # Select action (greedy, no exploration)
            action = agent.select_action(state, training=False)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track
            actions_taken.append(action)
            rewards_received.append(reward)
            balance_history.append(info['balance'])
            
            # Update
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Get performance summary
        perf = env.get_performance_summary()
        
        # Action distribution
        action_counts = pd.Series(actions_taken).value_counts().sort_index()
        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        
        # Compile results
        results = {
            'agent_name': agent_name,
            'initial_balance': self.initial_balance,
            'final_balance': perf['final_balance'],
            'total_return_pct': perf['total_return_pct'],
            'total_trades': perf['total_trades'],
            'winning_trades': perf['winning_trades'],
            'losing_trades': perf['losing_trades'],
            'win_rate': perf['win_rate'],
            'total_steps': steps,
            'balance_history': balance_history,
            'actions_taken': actions_taken,
            'rewards_received': rewards_received,
            'action_distribution': {action_names[i]: action_counts.get(i, 0) 
                                   for i in range(3)}
        }
        
        # Store results
        self.results[agent_name] = results
        
        # Print summary
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict):
        """Print backtest results"""
        print(f"\nInitial Capital: {results['initial_balance']:.2f} €")
        print(f"Final Capital: {results['final_balance']:.2f} €")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"\nAction Distribution:")
        for action, count in results['action_distribution'].items():
            pct = count / results['total_steps'] * 100
            print(f"  {action}: {count} ({pct:.1f}%)")
    
    def compare_agents(self, agents_results: List[Dict] = None):
        """
        Compare multiple agents
        
        Args:
            agents_results: List of results dicts (if None, use stored results)
        """
        if agents_results is None:
            agents_results = list(self.results.values())
        
        if not agents_results:
            print("No results to compare")
            return
        
        print(f"\n{'='*80}")
        print("AGENT COMPARISON")
        print(f"{'='*80}\n")
        
        # Create comparison table
        comparison = pd.DataFrame({
            'Agent': [r['agent_name'] for r in agents_results],
            'Return %': [r['total_return_pct'] for r in agents_results],
            'Final Balance': [r['final_balance'] for r in agents_results],
            'Trades': [r['total_trades'] for r in agents_results],
            'Win Rate %': [r['win_rate'] * 100 for r in agents_results]
        })
        
        print(comparison.to_string(index=False))
        
        # Best agent
        best_idx = comparison['Return %'].idxmax()
        best_agent = comparison.loc[best_idx, 'Agent']
        best_return = comparison.loc[best_idx, 'Return %']
        
        print(f"\n🏆 Best Agent: {best_agent} ({best_return:.2f}%)")
    
    def plot_results(
        self,
        agent_name: str = None,
        save_fig: bool = True,
        save_path: str = "models/saved/rl"
    ):
        """
        Plot backtest results
        
        Args:
            agent_name: Name of agent to plot (if None, plot all)
            save_fig: Whether to save figure
            save_path: Directory to save figure
        """
        if agent_name:
            results_to_plot = {agent_name: self.results[agent_name]}
        else:
            results_to_plot = self.results
        
        if not results_to_plot:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Balance Evolution
        for name, results in results_to_plot.items():
            axes[0, 0].plot(results['balance_history'], label=name, alpha=0.8)
        axes[0, 0].axhline(y=self.initial_balance, color='black', 
                          linestyle='--', alpha=0.3, label='Initial')
        axes[0, 0].set_title('Balance Evolution', fontsize=14)
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Balance ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Returns
        for name, results in results_to_plot.items():
            balance_hist = np.array(results['balance_history'])
            returns = (balance_hist / self.initial_balance - 1) * 100
            axes[0, 1].plot(returns, label=name, alpha=0.8)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0, 1].set_title('Cumulative Returns', fontsize=14)
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Action Distribution
        if len(results_to_plot) == 1:
            # Pie chart for single agent
            name = list(results_to_plot.keys())[0]
            results = results_to_plot[name]
            action_dist = results['action_distribution']
            
            labels = list(action_dist.keys())
            sizes = list(action_dist.values())
            colors = ['gray', 'green', 'red']
            
            axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                          startangle=90)
            axes[1, 0].set_title('Action Distribution', fontsize=14)
        else:
            # Bar chart for multiple agents
            for name, results in results_to_plot.items():
                action_dist = results['action_distribution']
                actions = list(action_dist.keys())
                counts = list(action_dist.values())
                x = np.arange(len(actions))
                axes[1, 0].bar(x, counts, alpha=0.7, label=name)
            axes[1, 0].set_title('Action Distribution', fontsize=14)
            axes[1, 0].set_xlabel('Action')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(actions)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Performance Metrics
        agents = list(results_to_plot.keys())
        returns = [results_to_plot[name]['total_return_pct'] for name in agents]
        win_rates = [results_to_plot[name]['win_rate'] * 100 for name in agents]
        
        x = np.arange(len(agents))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, returns, width, label='Return %', alpha=0.8)
        axes[1, 1].bar(x + width/2, win_rates, width, label='Win Rate %', alpha=0.8)
        axes[1, 1].set_title('Performance Metrics', fontsize=14)
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(agents, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            fig_path = save_path / "backtest_results.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {fig_path}")
        
        plt.show()
    
    def export_results(self, filepath: str = "models/saved/rl/backtest_results.csv"):
        """
        Export results to CSV
        
        Args:
            filepath: Path to save CSV file
        """
        if not self.results:
            print("No results to export")
            return
        
        # Create DataFrame
        data = []
        for name, results in self.results.items():
            data.append({
                'Agent': name,
                'Initial Balance': results['initial_balance'],
                'Final Balance': results['final_balance'],
                'Total Return %': results['total_return_pct'],
                'Total Trades': results['total_trades'],
                'Winning Trades': results['winning_trades'],
                'Losing Trades': results['losing_trades'],
                'Win Rate': results['win_rate'],
                'HOLD Actions': results['action_distribution']['HOLD'],
                'BUY Actions': results['action_distribution']['BUY'],
                'SELL Actions': results['action_distribution']['SELL']
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        
        print(f"\nResults exported to {filepath}")


def main():
    """Main backtesting script"""
    
    print("="*80)
    print("RL AGENT BACKTESTING - 2024")
    print("="*80)
    
    # Load test data
    print("\nLoading test data (2024)...")
    df_2024 = pd.read_parquet('data/processed/ml_dataset_2024.parquet')
    print(f"Test data: {len(df_2024)} rows")
    
    # Initialize backtester
    backtester = RLBacktester(initial_balance=10000.0)
    
    # Load trained agent
    print("\nLoading trained agent...")
    
    # Create agent with same architecture
    state_size = len([col for col in df_2024.columns 
                     if col not in ['target', 'timestamp_15m']]) + 3
    action_size = 3
    
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Load best model
    model_path = "models/saved/rl/best_agent.pth"
    try:
        agent.load(model_path)
        print(f"✓ Agent loaded from {model_path}")
    except FileNotFoundError:
        print(f"✗ Model not found: {model_path}")
        print("Please train the agent first using train_rl.py")
        return
    
    # Backtest
    results = backtester.backtest_agent(
        agent=agent,
        test_df=df_2024,
        agent_name="DQN Agent (Best)"
    )
    
    # Plot results
    backtester.plot_results(save_fig=True)
    
    # Export results
    backtester.export_results()
    
    print("\n" + "="*80)
    print("BACKTESTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()