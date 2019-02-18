#include <mpi.h>

namespace Scheduler {
	int proc_cnt; //currently serves as the resource count

	struct Task {
		String TaProl_cmds;
		int num_procs;
	};

	struct Job {
		int id;
		Task tsk;
		String System;
		Job next_job;
		private int jobid_cnt;
	
	};

	Class Job_WaitQueue {
		Job first_job;
		Job last_job;
		private int wait_cnt;
		
		Job get_next_job(int max_proc_cnt/*arguments are resource requirements*/) 
		{

		}

		void add_job(Job j)
		{

		}

		void remove_job(Job j)
		{

		}
	}

	Class Job_RunQueue {
		Job first_job;
		Job last_job;
		private int run_cnt;
		
		void add_job(Job j)
		{

		}

		void remove_job(Job j)
		{

		}
	}

	void Job_Submit(/*takes TaProl string and processor count*/)
	{
		//create task
		//create job
		//set job id
		//add job to wait queue
	}

	void Launch_Job( )
	{
		//remove job from wait queue
		//add job to running queue
		//update resource count
	}

	void Job_Done()
	{
		//remove job from run job queue
		//deallocate job and task
		//update resource count
	}
}
