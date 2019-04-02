# Reinforcement Learning Course By David Silver

Question [Reference](reference)

---
## Monte-Carlo Learning
Optimal value function V<sub>*</sub> with Monte-Carlo Agent running 100,000 episodes

**Value Function Update**
*V(S<sub>t</sub>)* <- *V(S<sub>t</sub>) + a (G<sub>t</sub> - V(S<sub>t</sub>))*

![Tri-Surface Plot][mcGraph]

---
## SARSA

MSE Per Lambda                  |  MSE Per Episode
:------------------------------:|:-------------------------:
![Point Plot][tdGraph1]  |  ![FacetGrid][tdGraph2]

---
## Linear Function Approximation

MSE Per Lambda                  |  MSE Per Episode
:------------------------------:|:-------------------------:
![Point Plot][lfaGraph1] |  ![FacetGrid][lfaGraph2]


[reference]: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
[mcGraph]: https://raw.githubusercontent.com/weiweitoo/easy21-rl/master/img/mc.png
[tdGraph1]: https://raw.githubusercontent.com/weiweitoo/easy21-rl/master/img/td1.png
[tdGraph2]: https://raw.githubusercontent.com/weiweitoo/easy21-rl/master/img/td2.png
[lfaGraph1]: https://raw.githubusercontent.com/weiweitoo/easy21-rl/master/img/lfa1.png
[lfaGraph2]: https://raw.githubusercontent.com/weiweitoo/easy21-rl/master/img/lfa2.png