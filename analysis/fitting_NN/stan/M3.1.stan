// 文件名: M3.1.stan
// 描述: 动态门控双学习率模型 - 并行运算维度转置版 (适配 tau 上限 10, theta 指数映射)

functions {
  real partial_log_lik_lpmf(array[] int seq, int start, int end, 
                            array[,] int choice, array[,] int reward, 
                            vector p0, vector inc_r1, vector inc_r0,
                            vector lr_pos1, vector lr_pos2, 
                            vector lr_neg1, vector lr_neg2, 
                            vector tau, vector theta, 
                            int nTrials, real k) {
    real p_log_lik = 0;
    
    int slice_len = end - start + 1;
    for (i in 1:slice_len) {
      int s = seq[i]; 
      
      real z = logit(p0[s]);  // belief state (logit space)
      real L = 0;             // evidence accumulator
      
      for (t in 1:nTrials) {
        real w  = inv_logit(k * (L - theta[s]));
        real ap = (1 - w) * lr_pos1[s] + w * lr_pos2[s];
        real an = (1 - w) * lr_neg1[s] + w * lr_neg2[s];

        // 使用 [t, s] 索引
        real eta = tau[s] * z;
        p_log_lik += bernoulli_logit_lpmf(choice[t, s] | eta);

        real p  = inv_logit(z);
        real pe = reward[t, s] - p;
        real a_used = (reward[t, s] == 1) ? ap : an;
        
        z += a_used * pe;
        L += reward[t, s] * inc_r1[s] + (1 - reward[t, s]) * inc_r0[s];
      }
    }
    return p_log_lik;
  }
}

data {
  int<lower=1> nSubjects;
  int<lower=1> nTrials;
  
  // 维度转置：[试次, 被试]
  array[nTrials, nSubjects] int<lower=0, upper=1> choice;   
  array[nTrials, nSubjects] int<lower=0, upper=1> reward;   
  array[1, nSubjects] real<lower=0, upper=1> mean_p;    
  real k; 
}

transformed data {
  vector<lower=0, upper=1>[nSubjects] p0;
  vector[nSubjects] inc_r1;
  vector[nSubjects] inc_r0;
  array[nSubjects] int seq; 
  
  real log_half = -log(2);
  for (s in 1:nSubjects){
    p0[s] = mean_p[1, s];
    inc_r1[s] = log_half - log(p0[s]);
    inc_r0[s] = log_half - log1m(p0[s]);
    seq[s] = s; 
  }
}

parameters {
  // ----- group-level raw means -----
  real lr_pos_mu_raw1;
  real lr_pos_mu_raw2;
  real lr_neg_mu_raw1;
  real lr_neg_mu_raw2;
  real tau_mu_raw;
  real theta_mu;

  // ----- group-level raw sds -----
  real<lower=0> lr_pos_sd_raw1;
  real<lower=0> lr_pos_sd_raw2;
  real<lower=0> lr_neg_sd_raw1;
  real<lower=0> lr_neg_sd_raw2;
  real<lower=0> tau_sd_raw;
  real<lower=0> theta_sd;

  // ----- subject raw effects -----
  vector[nSubjects] lr_pos_raw1;
  vector[nSubjects] lr_pos_raw2;
  vector[nSubjects] lr_neg_raw1;
  vector[nSubjects] lr_neg_raw2;
  vector[nSubjects] tau_raw;
  vector[nSubjects] theta_raw;
}

transformed parameters {
  vector<lower=0, upper=1>[nSubjects] lr_pos1;
  vector<lower=0, upper=1>[nSubjects] lr_pos2;
  vector<lower=0, upper=1>[nSubjects] lr_neg1;
  vector<lower=0, upper=1>[nSubjects] lr_neg2;
  vector<lower=0, upper=3>[nSubjects] tau;
  vector<lower=0>[nSubjects] theta; 

  for (s in 1:nSubjects) {
    lr_pos1[s] = Phi_approx(lr_pos_mu_raw1 + lr_pos_sd_raw1 * lr_pos_raw1[s]);
    lr_pos2[s] = Phi_approx(lr_pos_mu_raw2 + lr_pos_sd_raw2 * lr_pos_raw2[s]);
    lr_neg1[s] = Phi_approx(lr_neg_mu_raw1 + lr_neg_sd_raw1 * lr_neg_raw1[s]);
    lr_neg2[s] = Phi_approx(lr_neg_mu_raw2 + lr_neg_sd_raw2 * lr_neg_raw2[s]);
    
    // tau 上限调整为 10
    tau[s]     = 3 * Phi_approx(tau_mu_raw + tau_sd_raw * tau_raw[s]);
    
    // theta 指数映射无上界
    theta[s]   = 60 * Phi_approx(theta_mu + theta_sd * theta_raw[s]);
  }
}

model {
  // ----- Priors -----
  lr_pos_mu_raw1 ~ normal(0, 1);
  lr_pos_mu_raw2 ~ normal(0, 1);
  lr_neg_mu_raw1 ~ normal(0, 1);
  lr_neg_mu_raw2 ~ normal(0, 1);
  tau_mu_raw     ~ normal(0, 1);

  lr_pos_sd_raw1 ~ normal(0, 1);
  lr_pos_sd_raw2 ~ normal(0, 1);
  lr_neg_sd_raw1 ~ normal(0, 1);
  lr_neg_sd_raw2 ~ normal(0, 1);
  tau_sd_raw     ~ normal(0, 1);

  lr_pos_raw1 ~ normal(0, 1);
  lr_pos_raw2 ~ normal(0, 1);
  lr_neg_raw1 ~ normal(0, 1);
  lr_neg_raw2 ~ normal(0, 1);
  tau_raw     ~ normal(0, 1);

  theta_mu  ~ normal(0, 1);
  theta_sd  ~ normal(0, 1);
  theta_raw ~ normal(0, 1);

  // ----- Likelihood (使用 reduce_sum) -----
  target += reduce_sum(partial_log_lik_lpmf, seq, 1, 
                       choice, reward, p0, inc_r1, inc_r0,
                       lr_pos1, lr_pos2, lr_neg1, lr_neg2, 
                       tau, theta, nTrials, k);
}

generated quantities {
  // ----- Group-level readable params -----
  real<lower=0, upper=1> lr_pos_mu1 = Phi_approx(lr_pos_mu_raw1);
  real<lower=0, upper=1> lr_pos_mu2 = Phi_approx(lr_pos_mu_raw2);
  real<lower=0, upper=1> lr_neg_mu1 = Phi_approx(lr_neg_mu_raw1);
  real<lower=0, upper=1> lr_neg_mu2 = Phi_approx(lr_neg_mu_raw2);
  real<lower=0, upper=3> tau_mu    = 3 * Phi_approx(tau_mu_raw);
  real theta_mu_out = 60 * Phi_approx(theta_mu);

  // 维度转置输出 [试次, 被试]
  array[nTrials, nSubjects] real log_lik;
  array[nTrials, nSubjects] real p_time;         
  array[nTrials, nSubjects] real L_time;         
  array[nTrials, nSubjects] real w_time;         
  array[nTrials, nSubjects] real lr_pos_time;    
  array[nTrials, nSubjects] real lr_neg_time;    
  array[nTrials, nSubjects] int<lower=0, upper=1> y_pred;
  
  for (s in 1:nSubjects) {
    real z = logit(p0[s]);
    real L = 0;

    for (t in 1:nTrials) {
      real w  = inv_logit(k * (L - theta[s]));
      real ap = (1 - w) * lr_pos1[s] + w * lr_pos2[s];
      real an = (1 - w) * lr_neg1[s] + w * lr_neg2[s];

      real eta = tau[s] * z;

      log_lik[t, s]     = bernoulli_logit_lpmf(choice[t, s] | eta);
      y_pred[t, s]      = bernoulli_logit_rng(eta);

      p_time[t, s]      = inv_logit(z);
      L_time[t, s]      = L;
      w_time[t, s]      = w;
      lr_pos_time[t, s] = ap;
      lr_neg_time[t, s] = an;

      real p  = inv_logit(z);
      real pe = reward[t, s] - p;
      z += ((reward[t, s] == 1) ? ap : an) * pe;

      L += reward[t, s] * inc_r1[s] + (1 - reward[t, s]) * inc_r0[s];
    }
  }
}
