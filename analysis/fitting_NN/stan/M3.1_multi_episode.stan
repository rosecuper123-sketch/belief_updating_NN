// 文件名: M3.1_multi_episode.stan
// 描述: 动态门控双学习率模型 - 多 episode 联合拟合版
//
// 层级结构：group -> network(subject) -> episodes -> trials
//
// 核心原则：
// 1) nSubjects 表示独立训练得到的网络实例（network replicas）。
// 2) 同一网络的所有 episode 共享同一组认知参数：
//    lr_pos1, lr_pos2, lr_neg1, lr_neg2, tau, theta。
// 3) 每个 episode 的潜在状态独立重置：
//    z = logit(p0[s,e]); L = 0。
// 4) 所有 episode 的 trial-level log likelihood 联合进入同一网络参数的后验。
// 5) reduce_sum 仍然按 network/subject 维度并行。
// 6) 保持原模型的实际参数映射不变：tau ∈ (0,3)，theta ∈ (0,60)。

functions {
  real partial_log_lik_lpmf(
      array[] int seq,
      int start,
      int end,
      array[,,] int choice,
      array[,,] int reward,
      array[,] real p0,
      array[,] real inc_r1,
      array[,] real inc_r0,
      vector lr_pos1,
      vector lr_pos2,
      vector lr_neg1,
      vector lr_neg2,
      vector tau,
      vector theta,
      int nEpisodes,
      int nTrials,
      real k) {

    real p_log_lik = 0;

    int slice_len = end - start + 1;

    for (i in 1:slice_len) {
      int s = seq[i];

      // 同一 network 的多个 episode 共享参数，
      // 但每个 episode 都重新初始化潜在状态。
      for (e in 1:nEpisodes) {
        real z = logit(p0[s, e]);  // episode-specific initial belief
        real L = 0;                // episode-specific evidence accumulator

        for (t in 1:nTrials) {
          real w  = inv_logit(k * (L - theta[s]));
          real ap = (1 - w) * lr_pos1[s] + w * lr_pos2[s];
          real an = (1 - w) * lr_neg1[s] + w * lr_neg2[s];

          real eta = tau[s] * z;

          // choice[s, e, t]
          p_log_lik += bernoulli_logit_lpmf(choice[s, e, t] | eta);

          {
            real p = inv_logit(z);
            real pe = reward[s, e, t] - p;
            real a_used = (reward[s, e, t] == 1) ? ap : an;

            // 保持原模型的 latent-state update 不变
            z += a_used * pe;
          }

          // 保持原模型的 evidence accumulator update 不变
          L += reward[s, e, t] * inc_r1[s, e]
               + (1 - reward[s, e, t]) * inc_r0[s, e];
        }
      }
    }

    return p_log_lik;
  }
}

data {
  int<lower=1> nSubjects;  // 独立训练网络实例数
  int<lower=1> nEpisodes;  // 每个网络的测试 episode 数
  int<lower=1> nTrials;    // 每个 episode 的 trial 数

  // 统一使用 [network, episode, trial]
  array[nSubjects, nEpisodes, nTrials] int<lower=0, upper=1> choice;
  array[nSubjects, nEpisodes, nTrials] int<lower=0, upper=1> reward;

  // 每个 network × episode 的初始概率。
  // 即使目前同一 network 的各 episode 使用相同 mean_p，
  // 也建议在数据侧重复填入，以保持接口通用。
  array[nSubjects, nEpisodes] real<lower=0, upper=1> mean_p;

  real k;
}

transformed data {
  array[nSubjects, nEpisodes] real p0;
  array[nSubjects, nEpisodes] real inc_r1;
  array[nSubjects, nEpisodes] real inc_r0;

  array[nSubjects] int seq;

  real log_half = -log(2);

  for (s in 1:nSubjects) {
    seq[s] = s;

    for (e in 1:nEpisodes) {
      // logit(p0)、log(p0)、log1m(p0) 均要求严格位于 (0,1)
      if (mean_p[s, e] <= 0 || mean_p[s, e] >= 1) {
        reject("mean_p must be strictly between 0 and 1. Found mean_p[",
               s, ",", e, "] = ", mean_p[s, e]);
      }

      p0[s, e] = mean_p[s, e];
      inc_r1[s, e] = log_half - log(p0[s, e]);
      inc_r0[s, e] = log_half - log1m(p0[s, e]);
    }
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

  // ----- network/subject-level raw effects -----
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

  // 保持旧版实际映射不变
  vector<lower=0, upper=3>[nSubjects] tau;
  vector<lower=0, upper=60>[nSubjects] theta;

  for (s in 1:nSubjects) {
    lr_pos1[s] = Phi_approx(
      lr_pos_mu_raw1 + lr_pos_sd_raw1 * lr_pos_raw1[s]
    );
    lr_pos2[s] = Phi_approx(
      lr_pos_mu_raw2 + lr_pos_sd_raw2 * lr_pos_raw2[s]
    );
    lr_neg1[s] = Phi_approx(
      lr_neg_mu_raw1 + lr_neg_sd_raw1 * lr_neg_raw1[s]
    );
    lr_neg2[s] = Phi_approx(
      lr_neg_mu_raw2 + lr_neg_sd_raw2 * lr_neg_raw2[s]
    );

    // 注意：旧版实际范围是 (0, 3)，不是注释中的 10。
    tau[s] = 3 * Phi_approx(
      tau_mu_raw + tau_sd_raw * tau_raw[s]
    );

    // 注意：旧版实际是 60 * Phi_approx(...)，范围为 (0, 60)，不是 exp 映射。
    theta[s] = 60 * Phi_approx(
      theta_mu + theta_sd * theta_raw[s]
    );
  }
}

model {
  // ----- Priors: 保持原模型不变 -----
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

  // ----- Multi-episode joint likelihood -----
  // reduce_sum 继续按 network/subject 维度切片。
  target += reduce_sum(
    partial_log_lik_lpmf,
    seq,
    1,
    choice,
    reward,
    p0,
    inc_r1,
    inc_r0,
    lr_pos1,
    lr_pos2,
    lr_neg1,
    lr_neg2,
    tau,
    theta,
    nEpisodes,
    nTrials,
    k
  );
}

generated quantities {
  // ----- Group-level readable parameters -----
  real<lower=0, upper=1> lr_pos_mu1 = Phi_approx(lr_pos_mu_raw1);
  real<lower=0, upper=1> lr_pos_mu2 = Phi_approx(lr_pos_mu_raw2);
  real<lower=0, upper=1> lr_neg_mu1 = Phi_approx(lr_neg_mu_raw1);
  real<lower=0, upper=1> lr_neg_mu2 = Phi_approx(lr_neg_mu_raw2);

  // 保持旧版实际映射不变
  real<lower=0, upper=3> tau_mu = 3 * Phi_approx(tau_mu_raw);
  real<lower=0, upper=60> theta_mu_out = 60 * Phi_approx(theta_mu);

  // trial-level 输出：[network, episode, trial]
  array[nSubjects, nEpisodes, nTrials] real log_lik;
  array[nSubjects, nEpisodes, nTrials] real p_time;
  array[nSubjects, nEpisodes, nTrials] real L_time;
  array[nSubjects, nEpisodes, nTrials] real w_time;
  array[nSubjects, nEpisodes, nTrials] real lr_pos_time;
  array[nSubjects, nEpisodes, nTrials] real lr_neg_time;
  array[nSubjects, nEpisodes, nTrials] int<lower=0, upper=1> y_pred;

  // episode-level log likelihood：
  // 可用于把 episode 作为 held-out predictive unit 做 WAIC / PSIS-LOO。
  array[nSubjects, nEpisodes] real log_lik_episode;

  // 可选但很有用：network-level likelihood 汇总，
  // 用于检查每个 network 对总 likelihood 的贡献。
  vector[nSubjects] log_lik_subject;

  for (s in 1:nSubjects) {
    log_lik_subject[s] = 0;

    for (e in 1:nEpisodes) {
      real z = logit(p0[s, e]);
      real L = 0;

      log_lik_episode[s, e] = 0;

      for (t in 1:nTrials) {
        real w  = inv_logit(k * (L - theta[s]));
        real ap = (1 - w) * lr_pos1[s] + w * lr_pos2[s];
        real an = (1 - w) * lr_neg1[s] + w * lr_neg2[s];

        real eta = tau[s] * z;

        log_lik[s, e, t] =
          bernoulli_logit_lpmf(choice[s, e, t] | eta);

        log_lik_episode[s, e] += log_lik[s, e, t];
        log_lik_subject[s]    += log_lik[s, e, t];

        y_pred[s, e, t] = bernoulli_logit_rng(eta);

        // 保存“当前 trial 更新之前”的 latent quantities，
        // 与旧版 generated quantities 的时间定义保持一致。
        p_time[s, e, t]      = inv_logit(z);
        L_time[s, e, t]      = L;
        w_time[s, e, t]      = w;
        lr_pos_time[s, e, t] = ap;
        lr_neg_time[s, e, t] = an;

        {
          real p = inv_logit(z);
          real pe = reward[s, e, t] - p;
          real a_used = (reward[s, e, t] == 1) ? ap : an;

          z += a_used * pe;
        }

        L += reward[s, e, t] * inc_r1[s, e]
             + (1 - reward[s, e, t]) * inc_r0[s, e];
      }
    }
  }
}
