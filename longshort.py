class Longshort(Trading):
	def __init__(self, symbol, cash=10000, window=30, span=100, start=None):
		super().__init__(symbol, cash, window, span, start)

		self.action_space = spaces.Discrete(2)

	def _observation(self, cash, nown, p):
		flat_rate = 5 #5 dollars per transaction
		percent_rate = 0.005 # 0.5% fees
		short_fee = 0.05/300 #per-day fees

		cash_tot = cash + nown * p
        n_short = np.floor(cash_tot/p)
        n_long = n_short

		fee_short = int(nown == -n_short) * (flat_rate + abs(n_short - nown) * percent_rate)
        fee_long = int(nown == n_long) * (flat_rate + abs(n_long - nown) * percent_rate)
        while cash_tot-p*n_long-fee_long < 0:
        	n_long -= 1
        	fee_long = int(nown == n_long) * (flat_rate + abs(n_long - nown) * percent_rate)

        return np.array([
            [cash-p*n_long-fee_long, 	 n], # LONG
            [cash+p*n_short-fee_short, 	-n], # SHORT
        ])

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        state = self.state

        done = (self.i == 0)
        if not done:
            holdings = self._observations(state[1], state[2], self.price())[action]
            self.i -= 1
            self.state = np.hstack([[self.i, holdings[0], holdings[1]], self._history()])
            reward = self.eval(*holdings) - self.eval(state[1], state[2])
        else:
            holdings = [state[1] + state[2] * self.price, 0] # sell all
            self.state = np.hstack([[self.i, holdings[0], holdings[1]], self._history()])
            print('start', self.start, self.i, 'previous', (state[1], state[2]), 'current', holdings)
            reward = 0.0

        return np.array(self.state), reward, done, {}