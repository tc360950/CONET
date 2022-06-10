#ifndef ADAPTIVE_MH_H
#define ADAPTIVE_MH_H


template <class Real_t> class AdaptiveMH {
private:
	const Real_t epsilon = 0.0001;
	const Real_t initial_variance = 0.0001;
	const Real_t scaling = (1.0) * (1.0);
	const Real_t init_segment = 10;
	
	Real_t var{ initial_variance };
	Real_t average{ 0 };
	Real_t num_observations{ 0 };


public:
	AdaptiveMH<Real_t>() {}

	AdaptiveMH<Real_t>& operator=(const AdaptiveMH<Real_t> &g) {
		this->var = g.var;
		this->average = g.average;
		this->num_observations = g.num_observations;
		return *this;
	}

	Real_t get(Real_t x) {
		const Real_t old_average = average;
		average = average * num_observations / (num_observations + 1) + 1 / (num_observations + 1) * x;
		if (num_observations == 1) {
			var = x*x + old_average * old_average - (2)*average*average;
		} else if (num_observations > 0) {
			var = (num_observations - 1) / (num_observations)* var + (1 / (num_observations)) * (num_observations* old_average * old_average + x * x - (num_observations + 1)*average * average);
		}
		num_observations++;
		
		return num_observations < init_segment ? initial_variance : scaling * var + scaling * epsilon;
	}
};








#endif //ADAPTIVE_MH_H
