#include "scheduler.hpp"

namespace Scheduler {

	void Job_Submit(string cmds, int np)
	{
		//create task
		Task tsk;
		tsk.cmds=cmds;
		tsk.num_procs=np;

		//create job
		Job *jb=new Job(tsk, "NaN");
		
		//add job to wait queue
		waitq->add_job(jb,true,false);
	}

	int Launch_Job( )
	{
		//get next job
		Job* jb = waitq->get_next_job();
		int id = jb->id;
	
		//add job to running queue
		runq->add_job(jb,false,true);

		//remove job from wait queue
		waitq->remove_job(jb,false);
		
		return id;
	}

	void Job_Done(int jb_id)
	{
		Job *jb = runq->get_job_with_id(jb_id);

		//remove job from run job queue
		runq->remove_job(jb,true);
	}
}

