/* 
 * File:   kdbAutoSelectAutoSelect.cpp
 * Author: Administrator
 *
 * Created on 2018年3月21日, 下午6:51
 */
/*今天先整理一下思路。
 * 老师让我做的是三个式子的使用
 * 先实现一下general的做法，我觉得这个部分可以使用我之前做过的doubleLocal,因为那个时候那个部分的kdbAutoSelect的模型的构建是我自己写的！！！
 * 所以初步感觉在原先的基础上改写就可以了
 * 大致意思是：也是除了原文中有一个order，自己再设置一个数组，记录已经安排了父结点的部分。然后根绝已经安排父结点的个数，进行循环求接下来结点的
 * 最佳父结点（两个）
 * 根据老师给的三个公式。
 * 如果现在已经安排了父结点的为x1 x2 x3 前三个结点我感觉还是根据原先的来，那么现在给x4找父结点就变成了x1x2,x3 x1x3,x2 x2x3,x1三种情况的组合3*2/2
 * 现在安排好了x1x2x3x4给x5找父结点就变成了4*3/2=6种情况，分别是x1x2,x3x4 x1x3,x2x4 x1x4,x2x3 x2x3,x1x4 x2x4 x1x3 x3x4 x2x2 
 * 这边的编程情况要好好做一下
 */

#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "kdbAutoSelect.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

kdbAutoSelect::kdbAutoSelect() : pass_(1)
{
}

kdbAutoSelect::kdbAutoSelect(char*const*& argv, char*const* end) : pass_(1)
{ name_ = "KDB";

  // defaults
  k_ = 1;
  union_kdb_kdbAutoSelect = false;
  // get arguments
  while (argv != end) {
    if (*argv[0] != '-') {
      break;
    }
    else if (argv[0][1] == 'k') {
      getUIntFromStr(argv[0]+2, k_, "k");
    }else if (streq(argv[0] + 1, "un")) {
      union_kdb_kdbAutoSelect = true;
    }else {
      break;
    }

    name_ += argv[0];

    ++argv;
  }
}

kdbAutoSelect::~kdbAutoSelect(void)
{
}

void kdbAutoSelect::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class

class miCmpClass {
public:

    miCmpClass(std::vector<float> *m) {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*mi)[a] > (*mi)[b];
    }

private:
    std::vector<float> *mi;
};
void kdbAutoSelect::reset(InstanceStream &is) {
  instanceStream_ = &is;
  const unsigned int noCatAtts = is.getNoCatAtts();
  noCatAtts_ = noCatAtts;
  noClasses_ = is.getNoClasses();

  k_ = min(k_, noCatAtts_-1);  // k cannot exceed the real number of categorical attributes - 1
  
  // initialise distributions
  dTree_.resize(noCatAtts);
  parents_.resize(noCatAtts);

  for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
    parents_[a].clear();
    dTree_[a].init(is, a);
  }

  /*初始化各数据结构空间*/
  dist_1.reset(is);     //

  classDist_.reset(is);

  //pass_ = 1;
  trainingIsFinished_ = false;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
/*通过训练集来填写数据空间*/
void kdbAutoSelect::train(const instance &inst) {
    dist_1.update(inst);
    classDist_.update(inst);
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
void kdbAutoSelect::initialisePass() {
     assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
void kdbAutoSelect::finalisePass() {//现在的finalpass需要重写
    // calculate the mutual information from the xy distribution
    std::vector<float> mi;  
    getMutualInformation(dist_1.xxyCounts.xyCounts, mi);
//    for(int i=0;i<noCatAtts_;i++){
//           printf("%f\t",mi[i]);
//      }
//    printf("\n");
    // calculate the conditional mutual information from the xxy distribution
    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(dist_1.xxyCounts,cmi);
//    for(int i=0;i<noCatAtts_;i++){
//        for(int j=0;j<noCatAtts_;j++){
//            printf("%f\t",cmi[i][j]);
//        }
//        printf("\n");
//    }
//    printf("\n");
    
    //dist_.clear();

    // sort the attributes on MI with the class
    std::vector<CategoricalAttribute> order;
    std::vector<CategoricalAttribute> order_selected;//放置的是已经选择过的结点，也就是说可以做父结点的结点。
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
      order.push_back(a);
    }
    order_selected.clear();
    // assign the parents
    if (!order.empty()) {
      miCmpClass cmp(&mi);

      std::sort(order.begin(), order.end(), cmp);//我们依旧互信息排序来安排选择结点顺序
//      
//      for(int i=0;i<noCatAtts_;i++){
//           printf("%d\t%f\t",order[i],mi[order[i]]);
//      }
//      printf("\n");
      
      
      
      //放父结点之前的一次cmi的处理
      order_selected.push_back(order[0]);//排序后的前三个结点都不用考虑的 第一个没有父结点 第二个只有第一个父结点 第三个只有前两个父结点
      order_selected.push_back(order[1]);
      //parents_[order[1]][0]=order[0];
      parents_[order[1]].push_back(order[0]);
      order_selected.push_back(order[2]);
      //parents_[order[2]][0]=order[0];
      parents_[order[2]].push_back(order[0]);
      //parents_[order[2]][1]=order[1];
      parents_[order[2]].push_back(order[1]);
      while(order_selected.size()!=noCatAtts_){
          //double sum_cmi_best2=0;//最好的两个和
           double sum_cmi_best2=-1;
          double sum_subs=0;//要减去的部分，无关联的部分
          double sum_all=0;
          double sum_cmi_pre1=0;
          double sum_cmi_pre2=0;
          int I_,J_;//这是为了记录最好的两个父结点的下标
          //首先先计算条件互信息的和
          for(int i=0;i<order_selected.size();i++){
              sum_all+=cmi[order[order_selected.size()]][order[i]];//比如这个时候order 0 1 2 3结点选中了，size为4 
                                                           //所以这个时候要安排order[size]结点了
              
          }
          //printf("%f\n",sum_all);
          //任意两个组合成为待选节点的父结点 看哪种情况 order[order_selected.size()][i]+order[order_selected.size()][j]-sum_subs最大
          for(int i=0;i<order_selected.size();i++){
              sum_cmi_pre1=cmi[order[order_selected.size()]][order[i]];//两个条件互信息的和
              //printf("%f\t",sum_cmi_pre1);
              double sumT;//就是一个中间状态而已
              for(int j=i+1;j<order_selected.size();j++){
                  sum_cmi_pre2=sum_cmi_pre1+cmi[order[order_selected.size()]][order[j]];//这已经是两个条件互信息的和了 也就是老师给的第一个式子的和
                 // printf("%f\t",cmi[order[order_selected.size()]][order[j]]);
                 // printf("%f\t",sum_cmi_pre2);
                  sum_subs=sum_all-sum_cmi_pre2;//all-pre就是剩下的没有求和的部分 也就是老师给的第二个式子的和
                  //printf("%f\t",sum_subs);
                  sumT=sum_cmi_pre2-sum_subs;//做差之后的部分也就是老师给的第三个式子的和
                  if(sum_cmi_best2<=sumT){
                      sum_cmi_best2=sumT;
                      I_=i;//记录此时的最好的结点的order的位置
                      J_=j;
                  }
                   //printf("%d;%d,%f\n",i,j,sumT);//测试输出每一个循环中的每一种情况
              }
             
          }
         // printf("final");
          //printf("%d;%d,%f\n",I_,J_,sum_cmi_best2);
          //parents_[order[order_selected.size()]][0]=order[I_];
          parents_[order[order_selected.size()]].push_back(order[I_]);
          //parents_[order[order_selected.size()]][0]=order[J_];
           parents_[order[order_selected.size()]].push_back(order[J_]);
           
         order_selected.push_back(order[order_selected.size()]);  
      }
      
    }
    order.clear();
     //输出父节点
//            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() ; it != order.end(); it++) {
//                      printf("%d 's parent is  ", *it);
//                            for (unsigned int j = 0; j < parents_[*it].size(); j++) {
//                                unsigned int k= parents_[*it][j];
//                                 printf(" %u, ", k);
//                            }
//                            printf("\n");
//                        }               
 trainingIsFinished_ = true;

}

/// true iff no more passes are required. updated by finalisePass()
bool kdbAutoSelect::trainingIsFinished() {
  //return pass_ > 2;
     return trainingIsFinished_;
}

void kdbAutoSelect::classify(const instance& inst, std::vector<double> &posteriorDist) {
  // calculate the class probabilities in parallel
  // P(y)
    //printf("1\n");
    //不加un是全局的
  for (CatValue y = 0; y < noClasses_; y++) {
    posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
  }
    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents_[x1].size() == 0) {
                posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
            } else if (parents_[x1].size() == 1) {
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_[x1].size() == 2) {
                const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]), y);
                }
            }
        }
    }
    normalise(posteriorDist);
   // printf("2\n");
    if(union_kdb_kdbAutoSelect == true){
         parents_1.resize(noCatAtts_);
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            parents_1[a].clear();
        }

        std::vector<float> mi_loc;
        getMutualInformationloc(dist_1.xxyCounts.xyCounts, mi_loc, inst);

        // calculate the conditional mutual information from the xxy distribution
        crosstab<float> cmi_loc = crosstab<float>(noCatAtts_);
        getCondMutualInfloc(dist_1.xxyCounts, cmi_loc, inst);

        // sort the attributes on MI with the class
        std::vector<CategoricalAttribute> order1;
         std::vector<CategoricalAttribute> order_selected1;//放置的是已经选择过的结点，也就是说可以做父结点的结点。
         order_selected1.clear();
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            order1.push_back(a);
        }
        if(!order1.empty()){
             miCmpClass cmp(&mi_loc);

            std::sort(order1.begin(), order1.end(), cmp);
             //放父结点之前的一次cmi的处理
      order_selected1.push_back(order1[0]);//排序后的前三个结点都不用考虑的 第一个没有父结点 第二个只有第一个父结点 第三个只有前两个父结点
      order_selected1.push_back(order1[1]);
      //parents_[order[1]][0]=order[0];
      parents_1[order1[1]].push_back(order1[0]);
      order_selected1.push_back(order1[2]);
      //parents_[order[2]][0]=order[0];
      parents_1[order1[2]].push_back(order1[0]);
      //parents_[order[2]][1]=order[1];
      parents_1[order1[2]].push_back(order1[1]);
      while(order_selected1.size()<noCatAtts_){
          double sum_cmi_best2=-1;//最好的两个和
          double sum_subs=0;//要减去的部分，无关联的部分
          double sum_all=0;
          double sum_cmi_pre1=0;
          double sum_cmi_pre2=0;
          int I_,J_;//这是为了记录最好的两个父结点的下标
          //首先先计算条件互信息的和
          for(int i=0;i<order_selected1.size();i++){
              sum_all+=cmi_loc[order1[order_selected1.size()]][order1[i]];//比如这个时候order 0 1 2 3结点选中了，size为4 
                                                           //所以这个时候要安排order[size]结点了
              
          }
          //printf("%f\n",sum_all);
          //任意两个组合成为待选节点的父结点 看哪种情况 order[order_selected.size()][i]+order[order_selected.size()][j]-sum_subs最大
          for(int i=0;i<order_selected1.size();i++){
              sum_cmi_pre1=cmi_loc[order1[order_selected1.size()]][order1[i]];//两个条件互信息的和
              //printf("%f\t",sum_cmi_pre1);
              double sumT;//就是一个中间状态而已
              for(int j=i+1;j<order_selected1.size();j++){
                  sum_cmi_pre2=sum_cmi_pre1+cmi_loc[order1[order_selected1.size()]][order1[j]];//这已经是两个条件互信息的和了 也就是老师给的第一个式子的和
                 // printf("%f\t",cmi[order[order_selected.size()]][order[j]]);
                 // printf("%f\t",sum_cmi_pre2);
                  sum_subs=sum_all-sum_cmi_pre2;//all-pre就是剩下的没有求和的部分 也就是老师给的第二个式子的和
                  //printf("%f\t",sum_subs);
                  sumT=sum_cmi_pre2-sum_subs;//做差之后的部分也就是老师给的第三个式子的和
                  if(sum_cmi_best2<=sumT){
                      sum_cmi_best2=sumT;
                      I_=i;//记录此时的最好的结点的order的位置
                      J_=j;
                  }
                   //printf("%d;%d,%f\n",i,j,sumT);//测试输出每一个循环中的每一种情况
              }
             
          }
         // printf("final");
          //printf("%d;%d,%f\n",I_,J_,sum_cmi_best2);
          //parents_[order[order_selected.size()]][0]=order[I_];
          parents_1[order1[order_selected1.size()]].push_back(order1[I_]);
//          //parents_[order[order_selected.size()]][0]=order[J_];
           parents_1[order1[order_selected1.size()]].push_back(order1[J_]);
           
         order_selected1.push_back(order1[order_selected1.size()]);  
      }
        }
         
            //输出父节点
//            for (std::vector<CategoricalAttribute>::const_iterator it = order1.begin() ; it != order1.end(); it++) {
//                      printf("%d 's parent is  ", *it);
//                            for (unsigned int j = 0; j < parents_1[*it].size(); j++) {
//                                unsigned int k= parents_1[*it][j];
//                                 printf(" %u, ", k);
//                            }
//                            printf("\n");
//                        }  
         
         order1.clear();
         
          std::vector<double> posteriorDist1;
        posteriorDist1.assign(noClasses_, 0);
         for (CatValue y = 0; y < noClasses_; y++) {
            posteriorDist1[y] = dist_1.xxyCounts.xyCounts.p(y)* (std::numeric_limits<double>::max() / 2.0);
        }
        //printf("3\n");
        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
            for (CatValue y = 0; y < noClasses_; y++) {
                if (parents_1[x1].size() == 0) {
                    posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
                   // printf("test1\n");
                } else if (parents_1[x1].size() == 1) {
                    const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]));
                    if (totalCount1 == 0) {
                        posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                    }
                } else if (parents_1[x1].size() == 2) {
                    const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), parents_1[x1][1], inst.getCatVal(parents_1[x1][1]));
                    if (totalCount1 == 0) {
                        const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]));
                        if (totalCount2 == 0) {
                            posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                        } else {
                            posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), y);
                        }
                    } else {
                        posteriorDist1[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), parents_1[x1][1], inst.getCatVal(parents_1[x1][1]), y);
                    }
                }
            }
        }
//        //printf("4\n");
//           normalise(posteriorDist1);
//           //联合概率结合
        for (int classno = 0; classno < noClasses_; classno++) {
            posteriorDist[classno] += posteriorDist1[classno];
            posteriorDist[classno] = posteriorDist[classno] / 2;
        }  
    }
  // normalise the results
  //normalise(posteriorDist);
}





