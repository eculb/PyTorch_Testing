import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from collections import defaultdict

class SpendingDataset(Dataset):
    def __init__(self, csv_path):
        """
        Initialize the dataset from a financial CSV file
        """
        # Check if the file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"The file at {csv_path} was not found.")
        
        # Load the CSV data
        self.data_frame = pd.read_csv(csv_path)
        
        # Clean up column names (remove whitespace)
        self.data_frame.columns = [col.strip() for col in self.data_frame.columns]
        
        # Clean up amount column - convert to numeric
        if 'Amount' in self.data_frame.columns:
            self.data_frame['Amount'] = self.data_frame['Amount'].apply(
                lambda x: float(str(x).replace('$', '').replace(',', '')) if isinstance(x, str) else x
            )
        
        # Convert dates to datetime objects
        if 'Date' in self.data_frame.columns:
            self.data_frame['Date'] = pd.to_datetime(self.data_frame['Date'], errors='coerce')
        
        # Store numeric columns
        self.numeric_columns = self.data_frame.select_dtypes(include=[np.number]).columns.tolist()
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        return self.data_frame.iloc[idx]
    
    def analyze_spending(self):
        """Analyze spending patterns and return a summary"""
        spending_insights = {}
        
        # Total spending
        total_spending = self.data_frame['Amount'].sum()
        spending_insights['total_spending'] = total_spending
        
        # Spending by category
        category_spending = self.data_frame.groupby('Master Category')['Amount'].sum()
        spending_insights['category_spending'] = category_spending
        top_category = category_spending.idxmax()
        spending_insights['top_category'] = top_category
        spending_insights['top_category_amount'] = category_spending[top_category]
        
        # Spending by subcategory
        subcategory_spending = self.data_frame.groupby('Subcategory')['Amount'].sum()
        spending_insights['top_subcategory'] = subcategory_spending.idxmax()
        spending_insights['top_subcategory_amount'] = subcategory_spending.max()
        
        # Most frequent merchants
        if 'Payee' in self.data_frame.columns:
            merchant_counts = self.data_frame['Payee'].value_counts()
            spending_insights['most_frequent_merchant'] = merchant_counts.idxmax()
            spending_insights['most_frequent_merchant_count'] = merchant_counts.max()
            
            # Most expensive merchant
            merchant_spending = self.data_frame.groupby('Payee')['Amount'].sum()
            spending_insights['most_expensive_merchant'] = merchant_spending.idxmax()
            spending_insights['most_expensive_merchant_amount'] = merchant_spending.max()
        
        # Time analysis
        if 'Date' in self.data_frame.columns:
            # Spending by month
            self.data_frame['Month'] = self.data_frame['Date'].dt.month_name()
            monthly_spending = self.data_frame.groupby('Month')['Amount'].sum()
            if not monthly_spending.empty:
                spending_insights['monthly_spending'] = monthly_spending
                spending_insights['highest_spending_month'] = monthly_spending.idxmax()
                spending_insights['highest_spending_month_amount'] = monthly_spending.max()
            
            # Spending by day of week
            self.data_frame['Day_of_Week'] = self.data_frame['Date'].dt.day_name()
            daily_spending = self.data_frame.groupby('Day_of_Week')['Amount'].sum()
            if not daily_spending.empty:
                spending_insights['daily_spending'] = daily_spending
                spending_insights['highest_spending_day'] = daily_spending.idxmax()
                spending_insights['highest_spending_day_amount'] = daily_spending.max()
        
        # Payment method analysis
        if 'Payment Method' in self.data_frame.columns:
            payment_methods = self.data_frame.groupby('Payment Method')['Amount'].sum()
            spending_insights['payment_methods'] = payment_methods
            spending_insights['preferred_payment_method'] = payment_methods.idxmax()
        
        # Transaction frequency
        spending_insights['transaction_count'] = len(self.data_frame)
        
        # Average transaction amount
        spending_insights['average_transaction'] = self.data_frame['Amount'].mean()
        
        return spending_insights
    
    def generate_spending_summary(self):
        """Generate a descriptive paragraph about spending habits"""
        insights = self.analyze_spending()
        
        # Format currency values for better readability
        def format_currency(amount):
            return f"${amount:.2f}"
        
        summary = f"Based on your financial data, I can see that you've spent a total of {format_currency(insights['total_spending'])} "
        
        # Add time context if available
        if 'highest_spending_month' in insights:
            summary += f"with your highest spending in {insights['highest_spending_month']} "
            summary += f"({format_currency(insights['highest_spending_month_amount'])}). "
        else:
            summary += "across all recorded transactions. "
        
        # Add category information
        summary += f"Your top spending category is '{insights['top_category']}' "
        summary += f"at {format_currency(insights['top_category_amount'])}, "
        summary += f"with '{insights['top_subcategory']}' being your highest subcategory "
        summary += f"({format_currency(insights['top_subcategory_amount'])}). "
        
        # Add merchant information
        if 'most_frequent_merchant' in insights:
            summary += f"You shop most frequently at {insights['most_frequent_merchant']} "
            summary += f"({insights['most_frequent_merchant_count']} transactions), "
            summary += f"while your largest expenses are typically at {insights['most_expensive_merchant']} "
            summary += f"(total of {format_currency(insights['most_expensive_merchant_amount'])}). "
        
        # Add payment method if available
        if 'preferred_payment_method' in insights:
            summary += f"Your preferred payment method is {insights['preferred_payment_method']}. "
        
        # Add transaction patterns
        summary += f"On average, you spend {format_currency(insights['average_transaction'])} per transaction "
        
        # Add day of week pattern if available
        if 'highest_spending_day' in insights:
            summary += f"with {insights['highest_spending_day']} being your highest spending day of the week. "
        else:
            summary += "across all your purchases. "
        
        # Add personalized advice based on spending pattern
        if 'Auto/Transportation' in str(insights['category_spending'].index):
            auto_amount = insights['category_spending'].get('Auto/Transportation', 0)
            if auto_amount > 0.2 * insights['total_spending']:
                summary += "Your transportation costs are quite significant in your overall budget. "
                summary += "You might want to consider carpooling or public transportation to reduce these expenses. "
        
        return summary


def analyze_spending_habits(csv_path):
    """
    Main function to analyze spending habits from a CSV file
    """
    try:
        # Create the dataset
        dataset = SpendingDataset(csv_path)
        
        # Analyze the data
        print("\n===== SPENDING HABITS ANALYSIS =====")
        spending_summary = dataset.generate_spending_summary()
        print("\nYOUR SPENDING HABITS SUMMARY:")
        print(spending_summary)
        
        # Create a visualization of spending by category
        insights = dataset.analyze_spending()
        
        if 'category_spending' in insights and len(insights['category_spending']) > 0:
            plt.figure(figsize=(12, 6))
            insights['category_spending'].plot(kind='bar', color='skyblue')
            plt.title('Spending by Category')
            plt.xlabel('Category')
            plt.ylabel('Amount ($)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.grid(axis='y', alpha=0.3)
            plt.show()
        
        # If we have time-based data, show spending over time
        if 'Date' in dataset.data_frame.columns:
            # Group by date and sum amounts
            daily_spending = dataset.data_frame.groupby('Date')['Amount'].sum()
            
            if len(daily_spending) > 1:
                plt.figure(figsize=(12, 6))
                daily_spending.plot(kind='line', marker='o')
                plt.title('Spending Over Time')
                plt.xlabel('Date')
                plt.ylabel('Amount ($)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
        
        print("\nAnalysis complete!")
        return spending_summary
        
    except Exception as e:
        print(f"Error analyzing spending: {str(e)}")
        return f"Unable to analyze spending habits: {str(e)}"


if __name__ == "__main__":
    # Ask for CSV path if not provided
    csv_path = input("Enter the path to your financial CSV file: ")
    analyze_spending_habits(csv_path)