t|| d�j�r>|j|� n
|j|� W q tk
r�   y$tjt|| dd�j�r||j|� W n tk
r�   wY nX Y qX qW |r�|||| fV  x�|D ]�}y t|| |d�}	t|t	| d�}
W n8 tk
�r } z|d k	�r||� w�W Y d d }~X nX z>|�s0t
j|	t|
���rRt
j||�}t|
||||�E d H  W d t|
� X q�W |�sx|||| fV  d S )N)re   F)re   rd   )r<   rj   rk   r1   rl   rV   rH   ZS_ISLNKr5   ri   r
   rm   rY   rn   ro   )rq   Ztoppathr\   r]   rd   �namesr^   r_   r	   rp   ZdirfdZerrZdirpathr   r   r   rn   �  s<    



rn   c             G   s   t | |� dS )zpexecl(file, *args)

    Execute the executable file with argument list args, replacing the
    current process. N)�execv)�file�argsr   r   r   �execl
  s    rw   c             G   s    |d }t | |dd� |� dS )z�execle(file, *args, env)

    Execute the executable file with argument list args and
    environment env, replacing the current process. r?   N�����rx   )r=   )ru   rv   �envr   r   r   �execle  s    rz   c             G   s   t | |� dS )z�execlp(file, *args)

    Execute the executable file (which is searched for along $PATH)
    with argument list args, replacing the current process. N)�execvp)ru   rv   r   r   r   �execlp  s    r|   c             G   s    |d }t | |dd� |� dS )z�execlpe(file, *args, env)

    Execute the executable file (which is searched for along $PATH)
    with argument list args and environment env, replacing the current
    process. r?   Nrx   rx   )�execvpe)ru   rv   ry   r   r   r   �execlpe   s    r~   c             C   s   t | |� dS )z�execvp(file, args)

    Execute the executable file (which is searched for along $PATH)
    with argument list args, replacing the current process.
    args may be a list or tuple of strings. N)�_execvpe)ru   rv   r   r   r   r{   )  s    r{   c             C   s   t | ||� dS )z�