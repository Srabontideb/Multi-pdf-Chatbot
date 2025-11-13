css = '''
<style>
.chat-message {
    padding: 1.0rem; border-radius: 0.5rem; margin-bottom: 0.75rem; display: flex; gap: 0.75rem;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
  flex: 0 0 auto;
}
.chat-message .avatar img {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  flex: 1 1 auto;
  color: #fff;
  line-height: 1.45;
  word-break: break-word;
  overflow-wrap: anywhere;
  white-space: pre-wrap; /* keeps newlines while still wrapping */
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
  <div class="avatar">
    <img alt="bot avatar"
         referrerpolicy="no-referrer"
         src="https://drive.google.com/thumbnail?id=12QcNemRL1kfiXe5Rl98-llVx_F9WKEUF&sz=w96"
         onerror="this.onerror=null;this.src='https://i.ibb.co/5GzXkwq/user.png';"
         style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
  </div>
  <div class="message">{{MSG}}</div>
</div>
'''


user_template = '''
<div class="chat-message user">
  <div class="avatar">
    <img alt="user avatar"
         referrerpolicy="no-referrer"
         src="https://drive.google.com/thumbnail?id=1f2NBLTZSEnhHk3mABT0qFX3lPz82FW_p&sz=w80"
         onerror="this.onerror=null;this.src='https://i.ibb.co/5GzXkwq/user.png';">
  </div>
  <div class="message">{{MSG}}</div>
</div>
'''
