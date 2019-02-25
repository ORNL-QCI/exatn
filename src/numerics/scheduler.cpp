//#include <mpi.h>
#include <string>
using namespace std;

namespace Scheduler {
	int proc_cnt; //currently serves as the resource count

	struct Task {
		string TaProl_cmds;
		int num_procs;
	};

	struct Job {
		int id;
		Task tsk;
		string System;
		Job *next_job;

		Job(Task t, string s)
		{
			tsk=t;
			System=s;
			id=-1;
			next_job=NULL;
		}
	};

	class Job_Queue {
	private: 
		int jobid_cnt;
	public:
		Job *first_job;
		Job *last_job;
		
		Job_Queue()
		{
			jobid_cnt=0;
			first_job=NULL;
			last_job=NULL;
		}

		Job get_next_job(int max_proc_cnt/*arguments are resource requirements*/) 
		{

		}

		void add_job(Job *j, bool is_newj)
		{
			jobid_cnt+=1;
			if(is_newj) j->id=jobid_cnt;

			if(first_job == NULL)
			{
				first_job=j;
				last_job=j;
			}
			else if(first_job->next_job==NULL)
			{
				first_job->next_job=j;
				last_job=first_job->next_job;
			}
			else
			{
				last_job->next_job=j;
				last_job=last_job->next_job;
			}
		}

		void remove_job(Job j)
		{

		}
	};


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

	Job_Queue *waitq = new Job_Queue();
	Job_Queue *runq = new Job_Queue(); 
}
