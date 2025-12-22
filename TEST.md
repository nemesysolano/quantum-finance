## CLI Test ##
### Wavelets ###
```
 python3 -m qf base NSC --model wavelets
```

## Strategies ##

### Breakout Trading Strategy ###

It will only execute when the price has reached the breakout point. The breakout price must be at least 10% away from the breakout price (support or resistance).

### Reversal Trading Strategy ###

Reversal trading strategy using support and resistance lines (S & R lines) is exactly the counterpart of breakout trading strategy. Again, a 10% threshold is required to dispatch signal.

# Built Models #
barinbalance
probdiff
pricevol
wavelets