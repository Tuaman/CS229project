import numpy as np
import pandas as pd

class Environment:

	def __init__(self, total_trading_days = 300, init_money = 10000, 
		stock_name = 'amzn', first_trading_day = 30, num_history_price = 30):
		#read stock from file, return the array containing stock price
		self.stock_price = pd.read_csv('data/'+stock_name+'.us.csv')['Closes'].values 
		self.num_history_price = num_history_price #number of price history used to predict

		self.state_dict = {} 
		self.state_dict['today'] = first_trading_day
		self.state_dict['price_history'] = self.stock_price[first_trading_day - num_history_price:first_trading_day]
		self.state_dict['asset_allocation'] = [init_money, 0] #first element is the cash, the next ones are amount of stock hold
		self.state_dict['remaining_trading_day'] = total_trading_days
		self.state_dict['is_end'] = False
		self.state = self._make_state(self.state_dict)
		#start trading on the 30th day of the data because we want our model to consider price 30 days ago

	def _make_state(self, state_dict):
		state = []
		state.append(state_dict['asset_allocation'])
		state.append(state_dict['remaining_trading_day'])
		state.append(state_dict['price_history'])


		return state

	def step(self, action):

		asset = self.state_dict['asset_allocation']
		net_asset_before = asset[0] + self.state_dict['price_history'][-1] * asset[1]

		#check whether it is the last day of trading
		if self.state_dict['remaining_trading_day'] == 0
			self.state_dict['is_end'] = True
			self.re_allocate([1, 0])
			asset = self.state_dict['asset_allocation']
			net_asset_after = asset[0]
			reward = net_asset_after - net_asset_before
			return none, reward, self.state_dict['is_end']

		#if action = reallocate, then we reallocate according to our model
		if action['action'] == 're_allocate':
			self.re_allocate(action['new_allocation_ratio'])

		#Update the price history by shifting the price history by 1 timestep
		self.state_dict['price_history'] = self.stock_price[self.state_dict['today'] - self.num_history_price:self.state_dict['today']]
				
		#reduce the number of remaining trading day
		self.state_dict['today'] += 1
		self.state_dict['remaining_trading_day'] -= 1

		asset = self.state_dict['asset_allocation']
		net_asset_after = asset[0] + self.state_dict['price_history'][-1] * asset[1]
		reward = net_asset_after - net_asset_before

		self.state = self._make_state(self.state_dict)
		
		return self.state, reward, self.state_dict['is_end']

	def re_allocate(self, new_allocation_ratio):
		#unpack
		oa = self.state_dict['asset_allocation']
		today_stock_price = self.state_dict['price_history'][-1] #today_stock_price

		#total money 
		net_money = oa[0] + today_stock_price * oa[1]

		#calculate how much stock can we buy and buy accordingly
		num_stock = np.floor(money * new_allocation_ratio / today_stock_price)
		remaining_cash = net_money - num_stock * today_stock_price
		self.state_dict['asset_allocation'] = [remaining_cash, num_stock]



