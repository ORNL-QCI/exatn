//#include <mpi.h>
#include <iostream>
#include <string>
using namespace std;

namespace Scheduler {
	int proc_cnt; //currently serves as the resource count

	struct Task {
		string cmds;
		int num_procs;
	};

	struct Job {
		int id;
		Task tsk;
		string System;
		Job *next_job;

		Job (){ }

		Job(Task t, string s)
		{
			tsk=t;
			System=s;
			id=-1;
			next_job=NULL;
		}

		Job(Task t, string s, int i)
		{
			tsk=t;
			System=s;
			id=i;
			next_job=NULL;
		}
	};

	class Job_Queue {
	private: 
		int jobid_cnt;
		int qcnt;
	public:
		Job *first_job;
		Job *last_job;
		
		Job_Queue()
		{
			jobid_cnt=0;
			qcnt=0;
			first_job=NULL;
			last_job=NULL;
		}

		void print_queue()
		{
			Job *tmp = first_job;
			while(tmp != NULL)
			{
				cout << tmp->id << endl << tmp->tsk.cmds << tmp->tsk.num_procs << endl << tmp->System << endl << endl;
				tmp=tmp->next_job;
			}
		}

		Job* get_next_job() 
		{
			Job *temp=first_job;
			while(temp != NULL)
			{
				if(temp->tsk.num_procs <= proc_cnt) break;
				temp=temp->next_job;
			}
			
			return new Job(temp->tsk, temp->System, temp->id);
		}

		void add_job(Job *j, bool is_newj, bool is_runj)
		{
			qcnt++;
			jobid_cnt++;
			if(is_newj) j->id=jobid_cnt;
			if(is_runj) proc_cnt-=j->tsk.num_procs;

			if(first_job==NULL)
			{
				first_job=j;
				last_job=j;
				j=NULL;
			}
			else
			{
				last_job->next_job=j;
				last_job=j;
				j=NULL;
			}
		}

		void remove_job(Job *j, bool is_runj)
		{
			Job *cur=new Job();
			Job *prev=new Job();
			cur=first_job;
			while(cur != NULL)
			{
				if(cur->id == j->id && cur->tsk.cmds == j->tsk.cmds && cur->System == j->System && cur->tsk.num_procs == j->tsk.num_procs)
				{
					if(cur==first_job)
						first_job = first_job->next_job;
					else
					{
						prev->next_job = cur->next_job;
						if(prev->next_job==NULL)
							last_job=prev;
					}
					if(is_runj) proc_cnt+=j->tsk.num_procs;

					delete cur;
					qcnt--;
					break;
				}
				else
				{
					prev = cur;
					cur=cur->next_job;
				}
			}
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

