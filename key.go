package semcache

import (
	"crypto/sha256"
	"encoding/hex"
	"regexp"
	"strings"
)

// Redactor strips PII from query text before it is embedded, keyed, or stored,
// so secrets never enter the index. Patterns target synthetic Vietnamese PII
// shapes (CCCD/CMND, mobile numbers, bank cards/accounts, emails). Redaction
// runs before normalization. Order matters: more specific patterns first.
type Redactor struct {
	rules []redactRule
}

type redactRule struct {
	re    *regexp.Regexp
	token string
}

// Default ordering: email and grouped card numbers first (most specific), then
// 12-digit CCCD, then 0/+84 mobile numbers, then any remaining long digit run
// as a generic account number.
var defaultRedactRules = []redactRule{
	{regexp.MustCompile(`[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}`), "<EMAIL>"},
	{regexp.MustCompile(`\b\d{4}([ \-]\d{4}){3}\b`), "<CARD>"},
	{regexp.MustCompile(`\b\d{12}\b`), "<CCCD>"},
	{regexp.MustCompile(`(?:\+84|0)\d{9}\b`), "<PHONE>"},
	{regexp.MustCompile(`\b\d{6,}\b`), "<ACCT>"},
}

// NewRedactor returns a Redactor with the default Vietnamese-shaped rules.
func NewRedactor() *Redactor {
	return &Redactor{rules: defaultRedactRules}
}

// AddRule appends a custom redaction rule. Custom rules run after the defaults.
func (r *Redactor) AddRule(pattern, token string) error {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return err
	}
	r.rules = append(r.rules, redactRule{re: re, token: token})
	return nil
}

// Redact replaces every matched PII span with its placeholder token.
func (r *Redactor) Redact(text string) string {
	for _, rule := range r.rules {
		text = rule.re.ReplaceAllString(text, rule.token)
	}
	return text
}

// Canonicalize redacts PII then normalizes the text (lowercase, collapse
// whitespace, trim). This canonical form is what gets embedded, keyed, and
// matched — never the raw input.
func (r *Redactor) Canonicalize(text string) string {
	return normalize(r.Redact(text))
}

var wsRe = regexp.MustCompile(`\s+`)

// normalize lowercases, collapses internal whitespace, and trims.
func normalize(text string) string {
	return strings.TrimSpace(wsRe.ReplaceAllString(strings.ToLower(text), " "))
}

var tokenRe = regexp.MustCompile(`[\p{L}\p{N}<>]+`)

// tokenize splits canonical text into lowercase alphanumeric tokens. Placeholder
// tokens like <EMAIL> survive as units so verification can compare them.
func tokenize(text string) []string {
	return tokenRe.FindAllString(strings.ToLower(text), -1)
}

// deriveKey is the deterministic content key for a canonical query within a
// (namespace, epoch). Equal canonical text under the same namespace/epoch maps
// to the same key, so exact repeats overwrite rather than duplicate.
func deriveKey(namespace, epoch, canonical string) string {
	h := sha256.New()
	h.Write([]byte(namespace))
	h.Write([]byte{0})
	h.Write([]byte(epoch))
	h.Write([]byte{0})
	h.Write([]byte(canonical))
	return hex.EncodeToString(h.Sum(nil))
}
