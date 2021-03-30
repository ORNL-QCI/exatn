/** ExaTN::Numerics: Tensor contraction sequence optimizer factory
REVISION: 2020/04/29

Copyright (C) 2018-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2020 Oak Ridge National Laboratory (UT-Battelle) **/

#include "contraction_seq_optimizer_factory.hpp"

namespace exatn{

namespace numerics{

ContractionSeqOptimizerFactory::ContractionSeqOptimizerFactory()
{
 registerContractionSeqOptimizer("dummy",&ContractionSeqOptimizerDummy::createNew);
 registerContractionSeqOptimizer("heuro",&ContractionSeqOptimizerHeuro::createNew);
 registerContractionSeqOptimizer("greed",&ContractionSeqOptimizerGreed::createNew);
 registerContractionSeqOptimizer("metis",&ContractionSeqOptimizerMetis::createNew);
 registerContractionSeqOptimizer("cotengra",&ContractionSeqOptimizerCotengra::createNew);
}

void ContractionSeqOptimizerFactory::registerContractionSeqOptimizer(const std::string & name,
                                                                     createContractionSeqOptimizerFn creator)
{
 factory_map_[name] = creator;
 return;
}

std::unique_ptr<ContractionSeqOptimizer> ContractionSeqOptimizerFactory::createContractionSeqOptimizer(const std::string & name)
{
 auto it = factory_map_.find(name);
 if(it != factory_map_.end()) return (it->second)();
 return std::unique_ptr<ContractionSeqOptimizer>(nullptr);
}

std::shared_ptr<ContractionSeqOptimizer> ContractionSeqOptimizerFactory::createContractionSeqOptimizerShared(const std::string & name)
{
 return std::move(createContractionSeqOptimizer(name));
}

ContractionSeqOptimizerFactory * ContractionSeqOptimizerFactory::get()
{
 static ContractionSeqOptimizerFactory single_instance;
 return &single_instance;
}

} //namespace numerics

} //namespace exatn
