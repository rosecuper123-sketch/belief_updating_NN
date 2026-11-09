// 文件名: M1_parallel.stan
// 描述: 单学习率基线模型 (Phi_approx 参数化 + reduce_sum 多核并行)

functions {
  real partial_sum(array[] int slice_subj, int start, int end,     
                   int nTrials,
                   array[,] int choice,
                   array[,] int reward,
                   vector p0,
                   vector lr,
                   vector tau) {
    real pt_lp = 0.0;
    int slice_len = end - start + 1;
    
    for (i in 1:slice_len) {
      int s = slice_subj[i]; 
      real z = logit(p0[s]); 
      
      for (t in 1:nTrials) {
        real eta = tau[s] * z;
        pt_lp += bernoulli_logit_lpmf(choice[t, s] | eta);
        
        real p  = inv_logit(z);
        real pe = reward[t, s] - p;
        z += lr[s] * pe;
      }
    }
    return pt_lp;
  }
}

data {
  int<lower=1> nSubjects;
  int<lower=1> nTrials;
  
  // ★ 转置维度，对齐 R 端的 t()
  array[nTrials, nSubjects] int<lower=0, upper=1> choice;
  array[nTrials, nSubjects] int<lower=0, upper=1> reward;
  array[1, nSubjects] real<lower=0, upper=1> mean_p;
}

transformed data {
  vector<lower=0, upper=1>[nSubjects] p0; 
  array[nSubjects] int seq_subj; 
  for (s in 1:nSubjects) {
    p0[s] = mean_p[1, s];
    seq_subj[s] = s;
  }
}

parameters {
  real lr_mu_raw;  
  real tau_mu_raw;    
  real<lower=0> lr_sd_raw;
  real<lower=0> tau_sd_raw;
  vector[nSubjects] lr_raw;
  vector[nSubjects] tau_raw;
}

transformed parameters {
  vector<lower=0, upper=1>[nSubjects] lr;
  vector<lower=0, upper=3>[nSubjects] tau;

  for (s in 1:nSubjects) {
    lr[s]  = Phi_approx(lr_mu_raw + lr_sd_raw * lr_raw[s]);
    tau[s] = 3 * Phi_approx(tau_mu_raw + tau_sd_raw * tau_raw[s]);
  }
}

model {
  lr_mu_raw  ~ normal(0, 1);
  tau_mu_raw ~ normal(0, 1);
  lr_sd_raw  ~ normal(0, 1);
  tau_sd_raw ~ normal(0, 1);
  lr_raw     ~ normal(0, 1);
  tau_raw    ~ normal(0, 1);

  // 触发多核并行
  target += reduce_sum(partial_sum, seq_subj, 1,
                       nTrials, choice, reward, p0, lr, tau);
}

generated quantities {
  real<lower=0, upper=1> lr_mu = Phi_approx(lr_mu_raw);
  real<lower=0, upper=3> tau_mu = 3 * Phi_approx(tau_mu_raw);

  // ★ 维度对齐 R 端的提取逻辑
  array[nTrials, nSubjects] real log_lik;        
  for (s in 1:nSubjects) {
    real z = logit(p0[s]);
    for (t in 1:nTrials) {
      real eta = tau[s] * z;
      log_lik[t, s] = bernoulli_logit_lpmf(choice[t, s] | eta);
      
      real p  = inv_logit(z);
      real pe = reward[t, s] - p;
      z += lr[s] * pe;
    }
  }
}
