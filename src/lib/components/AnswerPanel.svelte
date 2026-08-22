<script lang="ts">
  let { answer }: { answer: string } = $props()
</script>

<p class="answer-text">{answer}</p>

<style>
  .answer-text {
    margin: 0;
    line-height: 1.6;
    font-size: 1rem;
  }
</style>
