B
    �a  �               @   s�   d dl Z d dlZd dlZd dlZd dlZd dlmZ d dlZe�e	�Z
eej�dd��Zejdeejj d�dd	��Zejde jd�dd
��ZdS )�    N)�List�DEBUG_COMPLETED_TIMEF� 皙�����?)�streamsc               s�  t j�� sdV  dS t j�� � |s*� g}n� fdd�|D �}dd� |D �}trnt jjdd�}� �|� t�� }t	�
d| ||� t �� }z
dV  W dt j�� }|� ks�t�tr�t�� }	x&t|�D ]\}
}||
 }|�|� q�W t �� }||ks�td��d	d� |D �}t	�
d
| |||� t j�� ��H x@t|��sbt�|�I dH  dd� |D �}t	�
d
| |||� �q$W W dQ R X t j�� }|� k�s�t�t�r�|	| d }d}x<t|�D ]0\}
}|�||
 �}|d|� d|d�d�7 }�q�W t	�d| |||� X dS )zPAsync context manager that waits for work to complete on given CUDA
    streams.Nc                s   g | ]}|r|n� �qS � r   )�.0�s)�stream_before_context_switchr   �8/home/buu/ltj/mmdetection/mmdet/utils/contextmanagers.py�
<listcomp>   s    zcompleted.<locals>.<listcomp>c             S   s   g | ]}t jjtd ��qS ))�enable_timing)�torch�cuda�Eventr   )r   �_r   r   r   r   "   s    T)r   z%s %s starting, streams: %sz)Unexpected is_grad_enabled() value changec             S   s   g | ]}|� � �qS r   )�query)r   �er   r   r   r   ?   s    z%s %s completed: %s streams: %sc             S   s   g | ]}|� � �qS r   )r   )r   r   r   r   r   r   E   s    i�  r   � z.2fz msz%s %s %.2f ms %s)r   r   �is_available�current_streamr   r   �record_event�time�	monotonic�logger�debug�is_grad_enabled�AssertionError�	enumerate�stream�all�asyncio�sleep�elapsed_time�info)�
trace_name�name�sleep_intervalr   Z
end_events�startZ	cpu_startZgrad_enabled_beforer   Zcpu_end�ir   �eventZgrad_enabled_afterZare_done�cpu_timeZstream_times_msr#   r   )r
   r   �	completed   sb    







r,   �
concurrentr   )�streamqueuec          
   C  s�   t j�� sdV  dS t j�� }t j�|��� | �� I dH }t|t jj�sLt�zXt j�|��B t	�
d|||� dV  t j�� }||ks�t�t	�
d|||� W dQ R X W d| ��  | �|� X W dQ R X dS )z�Run code concurrently in different streams.

    :param streamqueue: asyncio.Queue instance.

    Queue tasks define the pool of streams used for concurrent execution.
    Nz%s %s is starting, stream: %sz%s %s has finished, stream: %s)r   r   r   r   r   �get�
isinstance�Streamr   r   r   �	task_done�
put_nowait)r.   r%   r&   Zinitial_streamr   �currentr   r   r   r-   [   s$    




)r   r   r   N)r-   r   )r!   �
contextlib�logging�osr   �typingr   r   �	getLogger�__name__r   �bool�environr/   r   �asynccontextmanagerr   r1   r,   �Queuer-   r   r   r   r   �<module>   s    
   G                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    