/* Copyright (C) 2016 Guillaume Bonnoron

   This file is part of fplll. fplll is free software: you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation,
   either version 2.1 of the License, or (at your option) any later version.

   fplll is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with fplll. If not, see <http://www.gnu.org/licenses/>. */

//#include "test_utils.h"
#include <cstring>
#include <fplll/fplll.h>
#include <vector>
#include <set>
#include <math.h>
#include <gmp.h>
#include <mpfr.h>
#include <gmpxx.h>

#ifndef TESTDATADIR
#define TESTDATADIR ".."
#endif
using namespace std;
using namespace fplll;

/**
   @brief Test if CVP function returns correct vector.

   @param A              input lattice
   @param b              expected vector
   @return
*/

template <class ZT>
int test_cvp(unsigned int dim, ZZ_mat<ZT> &A, vector<Z_NR<mpz_t>> &target, vector<Z_NR<mpz_t>> &b,  const int method, mpz_t &the_last_solution)
{
  vector<Z_NR<mpz_t>> sol_coord;  // In the LLL-reduced basis
  vector<Z_NR<mpz_t>> solution;
  ZZ_mat<mpz_t> u;
  int status =
      lll_reduction(A, u, LLL_DEF_DELTA, LLL_DEF_ETA, LM_WRAPPER, FT_DEFAULT, 0, LLL_DEFAULT);
  if (status != RED_SUCCESS)
  {
    cerr << "LLL reduction failed: " << get_red_status_str(status) << endl;
    return status;
  }

  status = closest_vector(A, target, sol_coord, method);

  if (status != RED_SUCCESS)
  {
    cerr << "Failure: " << get_red_status_str(status) << endl;
    return status;
  }

  vector_matrix_product(solution, sol_coord, A);
  if (!solution.empty()) {
        // 获取 solution 最后一个元素的底层 mpz_t 数据
        const __mpz_struct* last_element_ptr = solution.back().get_data();
        mpz_set(the_last_solution, last_element_ptr); // 兼容 GMP 函数
    } else {
        // 处理空解情况（可选：设置默认值或报错）
        mpz_set_ui(the_last_solution, 0); // 示例：设为0
    }
    /*
      cerr << "A red" << endl << A << endl;
  cerr << "sol_coord : " << sol_coord << endl;
  cerr << "solution : " << solution << endl;
  cerr << "expected : " << b << endl;
  cerr << "target : " << target << endl;
  
    */

  bool correct = true;
  for (int i = 0; i < A.get_cols(); i++)
  {
    correct = correct && (solution[i] == b[i]);
  }
  if (!correct)
      return 1;
  return 0;
}


/**
   @brief Run CVP tests.

   @param argc             ignored
   @param argv             ignored
   @return
*/
void generate_prime(mpz_t p, unsigned int t) {
    // 计算最小对数 log2(p) ≥ (t²+15t)/4
    double log_p_min = (t*t + 15.0*t) / 4.0;
    // 转换为位长度并取上限
    unsigned long bit_len = static_cast<unsigned long>(std::ceil(log_p_min));
    
    // 计算最小p值: 2^bit_len
    mpz_t min_p;
    mpz_init(min_p);
    mpz_ui_pow_ui(min_p, 2, bit_len);
    
    // 生成下一个素数
    mpz_nextprime(p, min_p);
    mpz_clear(min_p);
}
// 步骤2: 计算噪声参数 l 和 R
void compute_noise_params(mpz_t R, unsigned long& l_val, const mpz_t p, unsigned int t) {
    // 使用MPFR高精度计算 log2(p)
    mpfr_t log2_p;
    mpfr_init2(log2_p, 256);  // 设置256位精度
    mpfr_set_z(log2_p, p, MPFR_RNDN);
    mpfr_log2(log2_p, log2_p, MPFR_RNDN);
    
    // 计算 l = ceil(t/4 + log2(p)/t + 1)
    double term1 = t / 4.0;
    double term2 = mpfr_get_d(log2_p, MPFR_RNDN) / t;
    l_val = static_cast<unsigned long>(std::ceil(term1 + term2 + 1.0));
    cout<<"l_val:"<< l_val<<endl;
    
    // 计算 R = ⌊p / 2^{l+1}⌋
    mpz_t divisor;
    mpz_init(divisor);
    mpz_ui_pow_ui(divisor, 2, l_val + 1);
    mpz_fdiv_q(R, p, divisor);  // 向下取整
    
    // 清理资源
    mpz_clear(divisor);
    mpfr_clear(log2_p);
}


int main(int argc, char *argv[])
{
  unsigned int t = 7;  // 门限值
  unsigned int n = 10;   // 份额数量
  cout<<"the number of the parties is "<<n<<",and the threshold is "<<t<<endl;
  // 初始化GMP整数
  mpz_t p, S;
  mpz_inits(p, S, NULL);
  cout<<"Generate the parameter of the Secret Sharing method: "<<endl;
  // 步骤1: 生成秘密共享参数
  generate_prime(p, t);
  gmp_printf("Prime p: %Zd\n", p);

  // 步骤2: 计算噪声参数
  mpz_t R;
  mpz_init(R);
  unsigned long l_val;
  compute_noise_params(R, l_val, p, t);
  gmp_printf("Noise bound R: %Zd\n", R);

  // 生成秘密 S ∈ Z_p^*
  gmp_randstate_t rand_state;
  gmp_randinit_default(rand_state);
  mpz_urandomm(S, rand_state, p);  // [0, p-1]
  if (mpz_sgn(S) == 0) mpz_set_ui(S, 1);  // 确保非零
  gmp_printf("The Secret shared is %Zd\n", S);
  cout<<"Generate the Secret Shares: "<<endl;
  //--- 使用FPLLL生成高质量随机噪声 ---
  fplll::RandGen::init();  // 初始化FPLLL随机源
  unsigned long seed=10086;
  fplll::RandGen::init_with_seed(10086);
  gmp_randstate_t fplll_rand;
  gmp_randinit_default(fplll_rand);
  // 存储份额和τ_i
  std::vector<mpz_class> taus;
  std::vector<mpz_class> ss;
  std::set<std::string> tau_set;  // 检测τ_i重复
    // 步骤3: 生成n个份额
  for (int i = 0; i < n; ++i) {
      mpz_t tau_i, s_i, e_i;
      mpz_inits(tau_i, s_i, e_i, NULL);
        
      // 生成不重复的τ_i ∈ Z_p^*
      while (true) {
          mpz_urandomm(tau_i, fplll_rand, p);   // [0, p-1]
          if (mpz_sgn(tau_i) == 0) continue;    // 排除0
            
          // 检测唯一性
          char tau_str[1000];
          gmp_sprintf(tau_str, "%Zd", tau_i);
          if (tau_set.insert(tau_str).second) break;
      }

      // 用FPLLL生成随机噪声 e_i ∈ (-R, R)
      mpz_t abs_e, bound;
      mpz_inits(abs_e, bound, NULL);
      mpz_sub_ui(bound, R, 1);  // R-1
        
      // 生成 [0, 2R-2] 的随机数
      mpz_mul_2exp(abs_e, bound, 1);
      mpz_add_ui(abs_e, abs_e, 1);  // 2R-1
      mpz_urandomm(e_i, fplll_rand, abs_e);
        
      // 转换为范围 [-R+1, R-1]
      mpz_sub(e_i, e_i, bound);
        
      // 计算 s_i = (S * τ_i + e_i) mod p
      mpz_mul(s_i, S, tau_i);
      mpz_add(s_i, s_i, e_i);
      mpz_mod(s_i, s_i, p);  // 取模
        
      // 保存份额 (τ_i, s_i)
      taus.emplace_back(tau_i);
      ss.emplace_back(s_i);
      std::cout<<"tau_"<<i<<": ";
      gmp_printf("%Zd,", tau_i);
      std::cout<<"ss_"<<i<<": ";
      gmp_printf("%Zd\n", s_i);
      // 清理临时变量
      mpz_clears(abs_e, bound, e_i, NULL);
    }
    
    // 清理资源
    //生成矩阵
    cout<<"Generate the latticebase: "<<endl;
    int dim=t+1;
    ZZ_mat<mpz_t> latticebase(dim,dim);
    //创建临时整数对象
    mpz_t  fp_p2,fp_R, fp_zero;
    mpz_init(fp_p2);
    mpz_init(fp_zero);
    mpz_mul(fp_p2,p,p);
    //gmp_printf("fp_p2: %Zd\n", fp_p2);
    // 1. 设置前t行（对角元素为p，其余为0）
    latticebase.gen_zero(dim,dim);
    for (int i = 0; i < t; i++) {
        for (int j = 0; j < dim; j++) {
            if (i == j) {
                //latticebase[i][j] = fp_p2;  // 对角线元素设为p
                latticebase.matrix[i][j]=fp_p2; 
            }
            else{
              continue;
            } 
        }
    }

    // 2. 设置最后一行
    mpz_t tau_mul_p_tmp;
    mpz_init(tau_mul_p_tmp);
    std::vector<mpz_class> tausmulp;
    for (size_t i = 0; i < t; i++)
    {
      mpz_mul(tau_mul_p_tmp,taus[i].get_mpz_t(),p);
      latticebase[t][i]=tau_mul_p_tmp;
    }
    latticebase[t][t]=R;
            //解决CVP问题之前检查
    cout<<"The lattice base is :"<<endl;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          gmp_printf("%Zd  ", latticebase[i][j]);
        }
        cout<<endl;
    }
    cout<<"Generate the target vector: "<<endl;
    //生成目标向量
    mpz_t adjusted;
    mpz_init(adjusted);
    vector<Z_NR<mpz_t>> target_vector;
    target_vector.reserve(dim);
    for (size_t i = 0; i < t; i++)
    {
      mpz_mul(adjusted, ss[i].get_mpz_t(), p);
      gmp_printf("%Zd ", adjusted);
      target_vector.push_back(adjusted);
    }
    target_vector.push_back(fp_zero);
    cout<<"target vector=";
    for (size_t i = 0; i < t+1; i++)
    {
          gmp_printf("%Zd ", target_vector[i]);
    }
    cout<<endl;
    cout<<"The time for calculate the Secret: "<<endl;
    //解决CVP问题
    vector<Z_NR<mpz_t>> target_u;
    target_u.reserve(dim);
    mpz_t result;
    mpz_init(result); // 必须初始化
    cout<<test_cvp(dim,latticebase,target_vector,target_u,CVPM_PROVED,result);
    //gmp_printf("Last solution: %Zd\n", result);
    mpz_div(result,result,R);
    //gmp_printf("After divition: %Zd\n", result);
    mpz_mod(result,result,p);
    gmp_printf("The Shared Secret is : %Zd\n", result);
    if (mpz_cmp(result,S))
    {
      cout<<"The secret reconstruct success!"<<endl;
    }
    mpz_clear(result); // 调用者负责清理
    mpz_clears(p, S, R, NULL);
    gmp_randclear(rand_state);
  return 0;
}
